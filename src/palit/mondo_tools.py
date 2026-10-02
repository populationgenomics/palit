"""MONDO tools the model calls while mapping an association onto a disease term.

* ``search_mondo``: ranked candidates for free-text queries over term labels and
  synonyms, or for an OMIM or Orphanet id.
* ``lookup_mondo``: a term's label, definition, synonyms, OMIM and Orphanet
  cross-references, and its neighbourhood in the tree: every ancestor up to the
  root, and descendants a few levels down.

Both run over a :class:`MondoIndex` built once from the local ``mondo.obo``. Only
human disease and disease-susceptibility terms that are not obsolete are
eligible: they are what search returns and what a mapping may name.
"""

import json
import logging
import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import jsonschema
import pronto
from anthropic.types import ToolParam

from palit.lookup_tools import ToolOutcome

logger = logging.getLogger(__name__)

DISEASE_ROOT = "MONDO:0000001"
SUSCEPTIBILITY_ROOT = "MONDO:0042489"
NON_HUMAN_ROOT = "MONDO:0005583"

XREF_PREFIXES = ("OMIM", "OMIMPS", "Orphanet")

MAX_QUERIES = 5
MAX_IDS = 5
SEARCH_RESULTS = 10
SEARCH_SYNONYMS = 8
SEARCH_DEFINITION_CHARS = 300
MAX_ANCESTORS = 60
DESCENDANT_LEVELS = 3
MAX_DESCENDANTS = 80

# Words that carry no weight in disease names; "type 7" and "7" name the same subtype.
STOP_WORDS = frozenset(
    {"a", "an", "and", "associated", "by", "due", "for", "in", "of", "or", "related"}
    | {"the", "to", "type", "with"}
)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_XREF_QUERY = re.compile(r"^\s*(OMIM|OMIMPS|Orphanet|ORPHA)\s*:\s*(\d+)\s*$", re.IGNORECASE)

SEARCH_TOOL: ToolParam = {
    "name": "search_mondo",
    "description": (
        "Searches MONDO human disease terms by label and synonym. Takes up to "
        f"{MAX_QUERIES} queries per call: search several phrasings, broader names and "
        "synonyms at once. A query may also be an OMIM or Orphanet id (e.g. OMIM:617439, "
        f"Orphanet:1234). For each query returns up to {SEARCH_RESULTS} candidates, best "
        "first: exact label matches, then exact synonym matches, then by word overlap. Each "
        "candidate has its id, label, the name that matched, a definition shortened to "
        f"{SEARCH_DEFINITION_CHARS} characters and up to {SEARCH_SYNONYMS} synonyms. Obsolete "
        "and non-human terms are never returned."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": MAX_QUERIES,
            }
        },
        "required": ["queries"],
        "additionalProperties": False,
    },
}

LOOKUP_TOOL: ToolParam = {
    "name": "lookup_mondo",
    "description": (
        f"Looks up MONDO terms by id (e.g. MONDO:0007915), up to {MAX_IDS} per call. For each "
        "returns the label, full definition, all synonyms with their scope, OMIM and Orphanet "
        "cross-references, every ancestor up to the root with its distance (1 = parent), and "
        f"descendants down to {DESCENDANT_LEVELS} levels with their depth (1 = child) and number "
        f"of children, at most {MAX_DESCENDANTS} listed (nearest levels first, groupings first "
        "within a level), with the full count per level. An "
        "obsolete term is marked as such with its replacement terms."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": MAX_IDS,
            }
        },
        "required": ["ids"],
        "additionalProperties": False,
    },
}

TOOLS: list[ToolParam] = [SEARCH_TOOL, LOOKUP_TOOL]

_VALIDATORS = {
    tool["name"]: jsonschema.Draft202012Validator(tool["input_schema"]) for tool in TOOLS
}


@dataclass(frozen=True)
class Synonym:
    text: str
    scope: str  # EXACT, RELATED, NARROW or BROAD


@dataclass(frozen=True)
class MondoTerm:
    id: str
    label: str
    definition: str  # empty when the term has none
    synonyms: tuple[Synonym, ...]
    xrefs: tuple[str, ...]  # OMIM, OMIMPS and Orphanet ids only
    parents: tuple[str, ...]  # MONDO is_a parents
    obsolete: bool
    replaced_by: tuple[str, ...]
    consider: tuple[str, ...]


@dataclass(frozen=True)
class _Name:
    """One searchable name of a term: its label or a synonym."""

    term_id: str
    text: str
    tokens: frozenset[str]
    key: tuple[str, ...]  # the name's tokens in order, for exact matching
    kind: int  # 0 label, 1 exact synonym, 2 other synonym


@dataclass(frozen=True)
class SearchHit:
    term: MondoTerm
    matched_name: str
    exact: bool


def tokens(text: str) -> list[str]:
    """Lowercased alphanumeric words of *text*, accents stripped, stop words dropped."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return [t for t in _NON_ALNUM.split(ascii_text.lower()) if t and t not in STOP_WORDS]


def _scope_kind(scope: str) -> int:
    return 1 if scope == "EXACT" else 2


class MondoIndex:
    """MONDO terms with their tree, a word index over names, and an xref index."""

    def __init__(self, terms: dict[str, MondoTerm], eligible: frozenset[str]) -> None:
        self._terms = terms
        self._eligible = eligible
        self._children: defaultdict[str, list[str]] = defaultdict(list)
        for term in terms.values():
            for parent in term.parents:
                self._children[parent].append(term.id)

        self._names: list[_Name] = []
        self._postings: defaultdict[str, list[int]] = defaultdict(list)
        self._by_xref: defaultdict[str, list[str]] = defaultdict(list)
        for term_id in sorted(eligible):
            term = terms[term_id]
            for xref in term.xrefs:
                self._by_xref[xref.upper()].append(term_id)
            names = [(term.label, 0)] + [(s.text, _scope_kind(s.scope)) for s in term.synonyms]
            seen: set[tuple[str, ...]] = set()
            for text, kind in names:
                key = tuple(tokens(text))
                if not key or key in seen:
                    continue
                seen.add(key)
                position = len(self._names)
                self._names.append(_Name(term_id, text, frozenset(key), key, kind))
                for token in set(key):
                    self._postings[token].append(position)
        # Inverse document frequency over terms rather than names, so a word repeated
        # across one term's synonyms counts once.
        term_counts = {
            token: len({self._names[p].term_id for p in positions})
            for token, positions in self._postings.items()
        }
        total = max(len(eligible), 1)
        self._idf = {token: math.log(1 + total / count) for token, count in term_counts.items()}
        self._unseen_idf = math.log(1 + total)

    @classmethod
    def from_ontology(cls, ontology: pronto.Ontology) -> "MondoIndex":
        terms: dict[str, MondoTerm] = {}
        for term in ontology.terms():
            if not term.id.startswith("MONDO:"):
                continue
            terms[term.id] = MondoTerm(
                id=term.id,
                label=term.name or term.id,
                definition=str(term.definition).strip() if term.definition else "",
                synonyms=tuple(
                    sorted(
                        (Synonym(s.description, s.scope or "RELATED") for s in term.synonyms),
                        key=lambda s: (_scope_kind(s.scope), s.text.lower()),
                    )
                ),
                xrefs=tuple(
                    sorted(x.id for x in term.xrefs if x.id.split(":", 1)[0] in XREF_PREFIXES)
                ),
                parents=tuple(
                    sorted(
                        p.id
                        for p in term.superclasses(distance=1, with_self=False)
                        if p.id.startswith("MONDO:")
                    )
                ),
                obsolete=term.obsolete,
                replaced_by=tuple(sorted(term.replaced_by.ids)),
                consider=tuple(sorted(term.consider.ids)),
            )
        index = cls(terms, frozenset())
        human = index.descendants_of([DISEASE_ROOT, SUSCEPTIBILITY_ROOT])
        non_human = index.descendants_of([NON_HUMAN_ROOT])
        eligible = frozenset(t for t in human - non_human if not terms[t].obsolete)
        logger.info("MONDO index: %d terms, %d eligible", len(terms), len(eligible))
        return cls(terms, eligible)

    def descendants_of(self, roots: Iterable[str]) -> set[str]:
        """*roots* present in the index and every term below them."""
        found: set[str] = set()
        stack = [root for root in roots if root in self._terms]
        while stack:
            term_id = stack.pop()
            if term_id in found:
                continue
            found.add(term_id)
            stack.extend(self._children.get(term_id, ()))
        return found

    def get(self, term_id: str) -> MondoTerm | None:
        return self._terms.get(term_id)

    def is_eligible(self, term_id: str) -> bool:
        return term_id in self._eligible

    # -----------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------

    def search(self, query: str, limit: int = SEARCH_RESULTS) -> list[SearchHit]:
        """Eligible terms for *query*, best first.

        An OMIM or Orphanet id returns the terms that cross-reference it. Text is
        ranked in tiers: the query's words equal the label, then an exact synonym,
        then another synonym; then by word overlap, weighted by inverse document
        frequency over the query's words, with ties broken by how little of the
        name is left unmatched.
        """
        xref = _XREF_QUERY.match(query)
        if xref is not None:
            prefix = "ORPHANET" if xref.group(1).upper() in ("ORPHA", "ORPHANET") else xref.group(1)
            ids = self._by_xref.get(f"{prefix.upper()}:{xref.group(2)}", [])
            return [SearchHit(self._terms[i], query.strip(), True) for i in ids[:limit]]

        key = tuple(tokens(query))
        if not key:
            return []
        query_tokens = set(key)
        weights = {t: self._idf.get(t, self._unseen_idf) for t in query_tokens}
        total_weight = sum(weights.values())

        best: dict[str, tuple[tuple[float, ...], _Name]] = {}
        positions = {p for token in query_tokens for p in self._postings.get(token, ())}
        for position in positions:
            name = self._names[position]
            if name.key == key:
                rank: tuple[float, ...] = (float(name.kind), 0.0, 0.0)
            else:
                shared = query_tokens & name.tokens
                coverage = sum(weights[t] for t in shared) / total_weight
                precision = len(shared) / len(name.tokens)
                rank = (3.0, -coverage, -precision)
            current = best.get(name.term_id)
            if current is None or (rank, name.kind) < (current[0], current[1].kind):
                best[name.term_id] = (rank, name)

        ordered = sorted(
            best.items(), key=lambda item: (item[1][0], self._terms[item[0]].label.lower())
        )
        return [
            SearchHit(self._terms[term_id], name.text, rank[0] < 3.0)
            for term_id, (rank, name) in ordered[:limit]
        ]

    def search_result(self, query: str) -> dict[str, Any]:
        """The model-facing result for one query."""
        return {
            "query": query,
            "candidates": [
                {
                    "id": hit.term.id,
                    "label": hit.term.label,
                    "matched_name": hit.matched_name,
                    "definition": _shorten(hit.term.definition, SEARCH_DEFINITION_CHARS),
                    "synonyms": [s.text for s in hit.term.synonyms[:SEARCH_SYNONYMS]],
                }
                for hit in self.search(query)
            ],
        }

    # -----------------------------------------------------------------------
    # Lookup
    # -----------------------------------------------------------------------

    def ancestors(self, term_id: str) -> list[tuple[str, int]]:
        """(ancestor id, shortest distance) for every MONDO ancestor, nearest first."""
        distances: dict[str, int] = {}
        frontier = list(self._terms[term_id].parents)
        distance = 1
        while frontier:
            next_frontier: list[str] = []
            for parent in frontier:
                if parent in distances or parent not in self._terms:
                    continue
                distances[parent] = distance
                next_frontier.extend(self._terms[parent].parents)
            frontier = next_frontier
            distance += 1
        return sorted(distances.items(), key=lambda item: (item[1], self._terms[item[0]].label))

    def child_count(self, term_id: str) -> int:
        """Number of eligible direct children."""
        return sum(child in self._eligible for child in self._children.get(term_id, ()))

    def descendant_levels(self, term_id: str, levels: int) -> list[list[str]]:
        """Eligible descendants by depth (index 0 = children), each term at its least depth.

        Within a level, terms with more children come first, so groupings are listed
        before single diseases when a level is cut short.
        """
        seen = {term_id}
        result: list[list[str]] = []
        frontier = [term_id]
        for _ in range(levels):
            level = sorted(
                {
                    child
                    for parent in frontier
                    for child in self._children.get(parent, ())
                    if child not in seen and child in self._eligible
                },
                key=lambda t: (-self.child_count(t), self._terms[t].label.lower()),
            )
            if not level:
                break
            seen.update(level)
            result.append(level)
            frontier = level
        return result

    def lookup_result(
        self,
        term_id: str,
        max_ancestors: int = MAX_ANCESTORS,
        levels: int = DESCENDANT_LEVELS,
        max_descendants: int = MAX_DESCENDANTS,
    ) -> dict[str, Any]:
        """The model-facing result for one id."""
        term = self._terms.get(term_id.strip())
        if term is None:
            return {"id": term_id, "status": "not found"}
        if term.obsolete:
            return {
                "id": term.id,
                "status": "obsolete",
                "label": term.label,
                "replaced_by": [self._brief(t) for t in term.replaced_by],
                "consider": [self._brief(t) for t in term.consider],
            }
        ancestors = self.ancestors(term.id)
        descendant_levels = self.descendant_levels(term.id, levels)
        listed: list[dict[str, Any]] = []
        for depth, level in enumerate(descendant_levels, start=1):
            for child in level[: max_descendants - len(listed)]:
                listed.append(
                    {**self._brief(child), "depth": depth, "children": self.child_count(child)}
                )
        result: dict[str, Any] = {
            "id": term.id,
            "status": "ok",
            "label": term.label,
            "definition": term.definition,
            "synonyms": [{"text": s.text, "scope": s.scope} for s in term.synonyms],
            "xrefs": {
                prefix: [x for x in term.xrefs if x.split(":", 1)[0] == prefix]
                for prefix in XREF_PREFIXES
                if any(x.split(":", 1)[0] == prefix for x in term.xrefs)
            },
            "ancestors": [{**self._brief(a), "distance": d} for a, d in ancestors[:max_ancestors]],
            "descendants": listed,
            "descendant_counts_by_depth": [len(level) for level in descendant_levels],
        }
        if len(ancestors) > max_ancestors:
            result["ancestors_not_listed"] = len(ancestors) - max_ancestors
        if not self.is_eligible(term.id):
            result["note"] = "Not a human disease term; it cannot be the answer."
        return result

    def _brief(self, term_id: str) -> dict[str, str]:
        term = self._terms.get(term_id)
        return {"id": term_id, "label": term.label if term is not None else "not a MONDO term"}


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class MondoToolRunner:
    """Executes ``search_mondo`` and ``lookup_mondo`` calls."""

    def __init__(self, index: MondoIndex) -> None:
        self._index = index

    def run(self, name: str, tool_input: dict[str, Any]) -> ToolOutcome:
        validator = _VALIDATORS.get(name)
        if validator is None:
            return ToolOutcome(json.dumps({"error": f"unknown tool {name}"}), is_error=True)
        errors = [e.message for e in validator.iter_errors(tool_input)]
        if errors:
            return ToolOutcome(
                json.dumps({"error": "invalid input", "details": errors[:5]}), is_error=True
            )
        if name == SEARCH_TOOL["name"]:
            results = [self._index.search_result(query) for query in tool_input["queries"]]
        else:
            results = [self._index.lookup_result(term_id) for term_id in tool_input["ids"]]
        return ToolOutcome(json.dumps({"results": results}), is_error=False)
