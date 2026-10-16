#!/usr/bin/env python3
"""Shared utilities for PanelApp criteria evaluation and integration."""

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from palit.gencc import MondoRef

# Target panel IDs
MENDELIOME_PANEL_ID = 137
INCIDENTALOME_PANEL_ID = 126
MITOCHONDRIAL_PANEL_ID = 203
REPEAT_DISORDERS_PANEL_ID = 3597
TARGET_PANEL_IDS = [
    MENDELIOME_PANEL_ID,
    INCIDENTALOME_PANEL_ID,
    MITOCHONDRIAL_PANEL_ID,
    REPEAT_DISORDERS_PANEL_ID,
]

# Panel ID to name mapping
PANEL_NAMES = {
    MENDELIOME_PANEL_ID: "Mendeliome",
    INCIDENTALOME_PANEL_ID: "Incidentalome",
    MITOCHONDRIAL_PANEL_ID: "Mitochondrial Disease",
    REPEAT_DISORDERS_PANEL_ID: "Repeat Disorders",
}

# An aggregated association's relation to PanelApp Australia's curation of the gene:
# curated with this MoI, curated only with another MoI, or not curated.
RelationStatus = Literal["existing", "new_disease", "new_moi"]
NEW_RELATION_STATUSES: frozenset[RelationStatus] = frozenset({"new_disease", "new_moi"})


class FindingKind(StrEnum):
    """What an association adds to the target panels' curation of a gene."""

    # A disease the target panels do not curate for the gene, which includes every
    # association of a gene on none of them
    NEW_DISEASE = "new_disease"
    NEW_MOI = "new_moi"  # highlighted new_moi associations only
    UPGRADE = "upgrade"  # an existing association rated higher in this corpus


@dataclass(frozen=True)
class Finding:
    """An association that the target panels lack or rate lower than this corpus does."""

    kind: FindingKind
    rating: int  # the association's corpus rating: 3 GREEN, 2 AMBER, 1 RED
    association_id: int
    # Upgrades only: the rating the corpus rating is above, see
    # generate_report.association_finding
    from_rating: int | None

    @property
    def badge(self) -> tuple[FindingKind, int, int | None]:
        """What the finding's badge shows; a gene's findings with the same badge share one."""
        return (self.kind, self.rating, self.from_rating)


# PanelApp criteria names
CriterionName = Literal["criterion_A", "criterion_B", "criterion_C", "criterion_D", "criterion_E"]
PANELAPP_CRITERIA: list[CriterionName] = [
    "criterion_A",
    "criterion_B",
    "criterion_C",
    "criterion_D",
    "criterion_E",
]


def validate_criteria_complete(criteria: list[dict[str, Any]]) -> bool:
    """Check that criteria contains exactly the 5 required entries A-E.

    Returns True if valid, False otherwise.
    """
    names = {c.get("name") for c in criteria}
    return names == set(PANELAPP_CRITERIA)


def criteria_object_to_list(entities: list[dict[str, Any]]) -> None:
    """Rewrite each entity's model-facing criteria object as the stored criteria list.

    The model answers with ``evidence_assessments`` as an object keyed
    ``criterion_A`` .. ``criterion_E``, so the output grammar enforces all five;
    everything downstream stores and reads a list of ``{"name": ..., ...}``
    entries in that order. Mutates *entities* in place.
    """
    for entity in entities:
        by_name: dict[str, dict[str, Any]] = entity["evidence_assessments"]
        entity["evidence_assessments"] = [
            {"name": name, **by_name[name]} for name in PANELAPP_CRITERIA
        ]


def validate_entities_criteria_complete(entities: list[dict[str, Any]]) -> bool:
    """Check that every disease entity has a complete 5-criterion evidence_assessments array.

    Returns True if all entities are valid, False otherwise.
    """
    return all(
        validate_criteria_complete(entity.get("evidence_assessments", [])) for entity in entities
    )


def get_entity_criterion(entity: dict[str, Any], name: CriterionName) -> dict[str, Any]:
    """Look up a criterion by name from a disease entity's evidence_assessments.

    Returns the criterion dict, or an empty dict if not found.
    """
    for c in entity.get("evidence_assessments", []):
        if c.get("name") == name:
            return dict(c)
    return {}


def validate_independent_family_counts(entities: list[dict[str, Any]]) -> bool:
    """Check each entity's independent_family_count is consistent with family_count:
    null exactly when family_count is null, otherwise 0 <= independent <= family_count.
    """
    for entity in entities:
        qualifying = entity.get("family_count")
        independent = entity.get("independent_family_count")
        if qualifying is None or independent is None:
            if qualifying is not None or independent is not None:
                return False
            continue
        if not 0 <= independent <= qualifying:
            return False
    return True


# Canonical mapping from evidence extraction enum to PanelApp long-form MoI
# Evidence extraction uses: "Monoallelic"|"Biallelic"|"Monoallelic_and_biallelic"|"X-linked"|"Mitochondrial"|"Other"|"NR"
ENUM_TO_PANELAPP_MOI = {
    "Monoallelic": "MONOALLELIC, autosomal or pseudoautosomal, NOT imprinted",
    "Biallelic": "BIALLELIC, autosomal or pseudoautosomal",
    "Monoallelic_and_biallelic": "BOTH monoallelic and biallelic, autosomal or pseudoautosomal",
    "X-linked": "X-LINKED: hemizygous mutation in males, monoallelic mutations in females may cause disease (may be less severe, later onset than males)",
    "Mitochondrial": "MITOCHONDRIAL",
    "Other": "Other",
    "NR": "",
}

PATERNAL_ALLELE_EXPRESSED = "imprinted, paternal allele expressed"
MATERNAL_ALLELE_EXPRESSED = "imprinted, maternal allele expressed"

# The phrases an aggregated association's inheritance_details may hold, joined by "; "
INHERITANCE_DETAILS_VOCABULARY: frozenset[str] = frozenset(
    {
        "reduced penetrance",
        "variable expressivity",
        "mosaic",
        PATERNAL_ALLELE_EXPRESSED,
        MATERNAL_ALLELE_EXPRESSED,
        "sex-limited",
    }
)

# The PanelApp MoI of a monoallelic gene whose associations name one imprinting direction.
# A maternally imprinted gene is silenced on the maternal allele, so the paternal one is expressed.
IMPRINTING_TO_PANELAPP_MOI = {
    PATERNAL_ALLELE_EXPRESSED: (
        "MONOALLELIC, autosomal or pseudoautosomal, maternally imprinted (paternal allele expressed)"
    ),
    MATERNAL_ALLELE_EXPRESSED: (
        "MONOALLELIC, autosomal or pseudoautosomal, paternally imprinted (maternal allele expressed)"
    ),
}
# The PanelApp MoI of a monoallelic gene whose associations name both imprinting directions
IMPRINTED_STATUS_UNKNOWN_PANELAPP_MOI = (
    "MONOALLELIC, autosomal or pseudoautosomal, imprinted status unknown"
)
# The imprinting direction each imprinted PanelApp MoI states
_PANELAPP_MOI_IMPRINTING = {moi: direction for direction, moi in IMPRINTING_TO_PANELAPP_MOI.items()}

# PanelApp MoIs that refine a mode of the extraction enum beyond its own PanelApp MoI
BOTH_BIALLELIC_MORE_SEVERE_PANELAPP_MOI = (
    "BOTH monoallelic and biallelic (but BIALLELIC mutations cause a more SEVERE disease form),"
    " autosomal or pseudoautosomal"
)
X_LINKED_BIALLELIC_FEMALES_PANELAPP_MOI = (
    "X-LINKED: hemizygous mutation in males, biallelic mutations in females"
)
# The MoI of a PanelApp entry that has none, as panel_entries.PanelEntry holds it
MOI_NOT_SPECIFIED = "not specified"

# The extraction enum mode that each MoI a PanelApp entry can hold names; None for one
# that names no mode
PANELAPP_MOI_MODES: dict[str, str | None] = {
    ENUM_TO_PANELAPP_MOI["Monoallelic"]: "Monoallelic",
    **dict.fromkeys(IMPRINTING_TO_PANELAPP_MOI.values(), "Monoallelic"),
    IMPRINTED_STATUS_UNKNOWN_PANELAPP_MOI: "Monoallelic",
    ENUM_TO_PANELAPP_MOI["Biallelic"]: "Biallelic",
    ENUM_TO_PANELAPP_MOI["Monoallelic_and_biallelic"]: "Monoallelic_and_biallelic",
    BOTH_BIALLELIC_MORE_SEVERE_PANELAPP_MOI: "Monoallelic_and_biallelic",
    ENUM_TO_PANELAPP_MOI["X-linked"]: "X-linked",
    X_LINKED_BIALLELIC_FEMALES_PANELAPP_MOI: "X-linked",
    ENUM_TO_PANELAPP_MOI["Mitochondrial"]: "Mitochondrial",
    "Unknown": None,
    "Other": None,
    MOI_NOT_SPECIFIED: None,
}


def inheritance_details_items(details: str) -> list[str]:
    """The "; "-separated items of an association's inheritance_details; none for ""."""
    return details.split("; ") if details else []


MondoMatch = Literal["gencc", "exact", "broader"]


@dataclass(frozen=True)
class MondoTerm:
    """The MONDO term stored on an association and how it was chosen."""

    mondo_id: str
    label: str
    match: MondoMatch
    rationale: str  # map-mondo's rationale; empty for a reused GenCC term
    obsolete: bool  # a reused GenCC term that the loaded MONDO release marks obsolete
    replaced_by: tuple[MondoRef, ...]  # MONDO's replacements for an obsolete term; often empty


@dataclass(frozen=True)
class PrefillAssociation:
    """What the PanelApp prefill takes from one of the gene's associations."""

    assessment: dict[str, Any]  # the stored association JSON, paper IDs in display form
    mondo: MondoTerm | None  # None until map-mondo has run for this association
    finding: Finding | None  # None for an association without a finding


@dataclass(frozen=True)
class PrefillEntry:
    """The gene's entry on a review form's panel, which the review extends.

    A review replaces the curator's evaluation of the entry with the form's rating,
    MoI, phenotypes and publications, so these cover the entry's curation as well as
    the described associations. Its comment is added to the earlier ones.
    """

    rating: int  # the gene's confidence level on the panel
    moi: str  # the entry's MoI, a key of PANELAPP_MOI_MODES
    phenotypes: tuple[str, ...]
    publications: tuple[int | str, ...]  # the entry's PMIDs and DOIs, in its order


FormType = Literal["add", "review"]


@dataclass
class PrefillData:
    """Prefill form data for PanelApp integration."""

    form_type: FormType
    panel_id: int
    hgnc_id: str  # "HGNC:8607" format for PanelApp
    rating: str  # "GREEN", "AMBER", "RED"
    moi: str  # Full PanelApp MoI string
    mode_of_pathogenicity: str | None
    # Semicolon-separated identifiers: a review's entry PMIDs and DOIs as the entry spells
    # them, then the associations' papers, by PMID where available and DOI otherwise
    publications: str
    # Semicolon-separated: a review's entry phenotypes as PanelApp lists them (free
    # text), then the associations' "label, MONDO:id" entries or bare disease names
    phenotypes: str
    comments: str  # what the described associations add, see prefill_comments


def derive_aggregate_moi(disease_entities: list[dict[str, Any]]) -> str:
    """Derive the overall inheritance mode from disease_entities.

    PanelApp requires a single MoI per gene, but our schema captures MoI per
    disease entity. This function aggregates across entities for PanelApp compatibility.

    Args:
        disease_entities: List of disease entity dicts, each with an inheritance_mode field

    Returns:
        The inheritance mode:
        - If all entities have the same mode -> that mode
        - If mixed Monoallelic + Biallelic -> Monoallelic_and_biallelic
        - If mixed with X-linked/Mitochondrial -> Other
        - If all NR -> NR
    """
    modes: set[str] = set()

    for entity in disease_entities:
        mode = entity.get("inheritance_mode")
        if mode and mode != "NR":
            modes.add(mode)

    # Drop "Other" when more specific modes are present (e.g. a somatic
    # mosaicism entity alongside several Monoallelic entities should not
    # force the aggregate to "Other").
    specific_modes = modes - {"Other"}
    if specific_modes:
        modes = specific_modes

    # Derive mode
    if not modes:
        return "NR"

    if len(modes) == 1:
        return modes.pop()

    # Multiple modes - check for mono+bi combination
    if modes == {"Monoallelic", "Biallelic"}:
        return "Monoallelic_and_biallelic"

    # If Monoallelic_and_biallelic is already in there with either mono or bi
    if "Monoallelic_and_biallelic" in modes:
        remaining = modes - {"Monoallelic_and_biallelic", "Monoallelic", "Biallelic"}
        if not remaining:
            return "Monoallelic_and_biallelic"

    # Mixed with X-linked, Mitochondrial, or Other
    return "Other"


def derive_panelapp_moi(associations: list[dict[str, Any]]) -> str:
    """The gene's PanelApp MoI string, aggregated over its associations.

    A gene whose aggregate mode is Monoallelic takes its imprinting status from
    the inheritance_details of its Monoallelic associations (the only ones that
    yield that mode): one imprinting direction gives that direction's PanelApp
    MoI, both give "imprinted status unknown", none gives "NOT imprinted".
    Imprinting phrases on associations of other modes don't change the MoI.
    """
    mode = derive_aggregate_moi(associations)
    if mode != "Monoallelic":
        return ENUM_TO_PANELAPP_MOI[mode]
    directions = {
        item
        for association in associations
        if association["inheritance_mode"] == "Monoallelic"
        for item in inheritance_details_items(association["inheritance_details"])
        if item in IMPRINTING_TO_PANELAPP_MOI
    }
    if not directions:
        return ENUM_TO_PANELAPP_MOI["Monoallelic"]
    if len(directions) == 1:
        return IMPRINTING_TO_PANELAPP_MOI[directions.pop()]
    return IMPRINTED_STATUS_UNKNOWN_PANELAPP_MOI


def derive_review_moi(entry_moi: str, associations: list[dict[str, Any]]) -> str:
    """A review form's MoI: the entry's MoI on the panel combined with *associations*.

    The entry's MoI counts as one more association, of the mode it names (see
    PANELAPP_MOI_MODES) and with the imprinting direction it states; an MoI that names
    no mode counts for nothing. The combination follows derive_panelapp_moi. When the
    associations leave the MoI the entry alone yields, the entry's own MoI stands,
    which keeps what the extraction enum cannot express: "imprinted status unknown",
    a more severe biallelic form, biallelic variants in females. So does an entry's
    "Unknown" or "Other" when no association reports a mode either.
    """
    mode = PANELAPP_MOI_MODES[entry_moi]
    if mode is None:
        combined = derive_panelapp_moi(associations)
        unreported = combined == ENUM_TO_PANELAPP_MOI["NR"]
        return entry_moi if unreported and entry_moi != MOI_NOT_SPECIFIED else combined
    entry = {
        "inheritance_mode": mode,
        "inheritance_details": _PANELAPP_MOI_IMPRINTING.get(entry_moi, ""),
    }
    combined = derive_panelapp_moi([entry, *associations])
    return entry_moi if combined == derive_panelapp_moi([entry]) else combined


@dataclass(frozen=True)
class PrefillDisease:
    """How the prefill names an association's disease."""

    name: str
    mondo_id: str | None  # None for an association without a MONDO term


def prefill_disease(association: PrefillAssociation) -> PrefillDisease:
    """The association's disease as the prefill names it.

    A reused GenCC term and an exact match are named by their MONDO label, a broader
    match by the proposed disease name. A reused GenCC term that MONDO obsoleted with
    exactly one replacement gives the replacement instead; with no or several
    replacements it stays as submitted. An association without a MONDO term is named
    by its proposed disease name.
    """
    mondo = association.mondo
    if mondo is None or mondo.match == "broader":
        proposed_name: str = association.assessment["proposed_disease_name"]
        return PrefillDisease(proposed_name, None if mondo is None else mondo.mondo_id)
    if mondo.obsolete and len(mondo.replaced_by) == 1:
        replacement = mondo.replaced_by[0]
        return PrefillDisease(replacement.label, replacement.mondo_id)
    return PrefillDisease(mondo.label, mondo.mondo_id)


def prefill_phenotype(association: PrefillAssociation) -> str:
    """The association's entry in the prefill's phenotype field: ``name, MONDO:id``, or the
    name alone without a MONDO term (see prefill_disease)."""
    disease = prefill_disease(association)
    return disease.name if disease.mondo_id is None else f"{disease.name}, {disease.mondo_id}"


# A MONDO ID in a PanelApp phenotype string
_MONDO_ID = re.compile(r"MONDO:\d{7}")


# How a prefill comment's heading names each MoI of the extraction enum
HEADING_MOI = {
    "Monoallelic": "Monoallelic",
    "Biallelic": "Biallelic",
    "Monoallelic_and_biallelic": "Monoallelic and biallelic",
    "X-linked": "X-linked",
    "Mitochondrial": "Mitochondrial",
    "Other": "other MoI",
    "NR": "MoI not reported",
}

# The prefill's name of each confidence level. PanelApp's review form rates Green,
# Amber or Red only, so confidence 0 (not rated) is Red, as panel_entries reads it.
PREFILL_RATINGS = {0: "Red", 1: "Red", 2: "Amber", 3: "Green"}


def prefill_comment_heading(association: PrefillAssociation, form_type: FormType) -> str:
    """The heading of the association's section of the prefill comment.

    It names the disease as prefill_disease does, its first letter capitalised and
    without its MONDO ID, with the MoI (see HEADING_MOI) and the association's rating:
    "<Disease> (<MoI>, <Rating>)". On a review form an association with a finding is
    headed by its kind: "New disease: ", "New MoI: " or "Upgrade from <rating>: ".
    """
    assessment = association.assessment
    name = prefill_disease(association).name
    disease = name[0].upper() + name[1:]
    moi = HEADING_MOI[assessment["inheritance_mode"]]
    rating = PREFILL_RATINGS[calculate_association_rating(assessment)]
    heading = f"{disease} ({moi}, {rating})"
    finding = association.finding
    if form_type == "add" or finding is None:
        return heading
    match finding.kind:
        case FindingKind.NEW_DISEASE:
            kind = "New disease"
        case FindingKind.NEW_MOI:
            kind = "New MoI"
        case FindingKind.UPGRADE:
            if finding.from_rating is None:
                raise ValueError(
                    f"Upgrade finding of association {finding.association_id} has no from_rating"
                )
            kind = f"Upgrade from {PREFILL_RATINGS[finding.from_rating]}"
    return f"{kind}: {heading}"


def prefill_comments(form_type: FormType, associations: list[PrefillAssociation]) -> str:
    """The prefill's comment: what the described associations add to PanelApp's curation.

    PanelApp shows the comment publicly, next to the curator's earlier ones, so it
    describes the associations alone. Each association gets a section of its heading
    (see prefill_comment_heading) and its summary, the sections separated by a blank
    line. An add form that describes one association gives its summary alone.
    """
    if form_type == "add" and len(associations) == 1:
        summary: str = associations[0].assessment["summary"]
        return summary
    return "\n\n".join(
        f"{prefill_comment_heading(a, form_type)}\n{a.assessment['summary']}" for a in associations
    )


def prefill_publications(
    entry_publications: tuple[int | str, ...],
    dois: list[str],
    doi_to_pmid: dict[str, int | None],
) -> list[str]:
    """The prefill's publications: *entry_publications*, then the papers of *dois*.

    A paper of *dois* is listed by its PMID when it has one, by its DOI otherwise. Each
    paper is listed once, where it first appears: a DOI is the same paper as the PMID
    *doi_to_pmid* gives it, and DOIs match in any letter case.
    """
    pmid_by_doi = {doi.lower(): pmid for doi, pmid in doi_to_pmid.items()}

    def paper(identifier: int | str) -> int | str:
        if isinstance(identifier, int):
            return identifier
        doi = identifier.lower()
        pmid = pmid_by_doi.get(doi)
        return doi if pmid is None else pmid

    report_publications = [doi if (pmid := doi_to_pmid[doi]) is None else pmid for doi in dois]
    listed: dict[int | str, str] = {}
    for identifier in (*entry_publications, *report_publications):
        listed.setdefault(paper(identifier), str(identifier))
    return list(listed.values())


def prepare_prefill_data(
    hgnc_id: int,
    associations: list[PrefillAssociation],
    moi_associations: list[dict[str, Any]],
    panel_id: int,
    entry: PrefillEntry | None,
    doi_to_pmid: dict[str, int | None],
    rated: bool = True,
) -> PrefillData:
    """One PanelApp prefill for the gene, as the union over *associations*.

    *associations* are the ones the prefill describes, which the caller chooses
    (the report's choice is generate_report.prefill_associations). A gene not
    on *panel_id* (no *entry*) gets an add form rated with the top rating of
    *associations*. A gene on it gets a review form of its *entry* there, rated with
    the higher of the gene's rating on the panel and that top rating: the corpus holds
    only the papers this run read, so a lower rating is no downgrade recommendation.
    When not *rated*, the associations do not count toward the rating, a review form
    keeps the gene's rating on the panel, and the comment is empty: the target panels
    have reviewed every paper it would describe.

    A review replaces the curator's evaluation of the entry, so its fields describe
    the whole gene. Its MoI combines the entry's MoI with *moi_associations*, every
    association of the gene and not only the described ones (see derive_review_moi);
    its phenotypes and publications are the entry's, then those of *associations*. An
    add form's MoI is aggregated over *moi_associations* (see derive_panelapp_moi),
    and its phenotypes and publications are those of *associations*. Phenotypes and
    publications follow the order of *associations*, each listed once; phenotypes
    match after surrounding whitespace is stripped, publications as prefill_publications
    describes. A phenotype of *associations* whose MONDO ID an entry phenotype holds is
    not listed. The comment describes the associations alone (see prefill_comments).

    Args:
        hgnc_id: HGNC ID (integer) of the gene
        associations: The associations the prefill describes, in report order
        moi_associations: Every association of the gene, for the MoI
        panel_id: Target panel ID
        entry: The gene's entry on *panel_id*; None when it is not on it
        doi_to_pmid: PMID (None when the paper has none) of every DOI the associations cite
        rated: Whether the associations' top rating counts toward the form's rating

    Returns:
        PrefillData object ready for form rendering
    """
    assessments = [a.assessment for a in associations]
    top_rating = calculate_gene_rating(assessments)
    form_type: FormType
    if entry is None:
        if not rated:
            raise ValueError("An add form is rated by its associations")
        form_type, rating = "add", top_rating
        moi = derive_panelapp_moi(moi_associations)
        entry_phenotypes: tuple[str, ...] = ()
        entry_publications: tuple[int | str, ...] = ()
    else:
        form_type = "review"
        rating = max(entry.rating, top_rating) if rated else entry.rating
        moi = derive_review_moi(entry.moi, moi_associations)
        entry_phenotypes, entry_publications = entry.phenotypes, entry.publications

    entry_mondo_ids = {m for phenotype in entry_phenotypes for m in _MONDO_ID.findall(phenotype)}
    phenotypes = dict.fromkeys(
        phenotype.strip()
        for phenotype in (
            *entry_phenotypes,
            *(
                prefill_phenotype(a)
                for a in associations
                if prefill_disease(a).mondo_id not in entry_mondo_ids
            ),
        )
    )
    publications = prefill_publications(
        entry_publications,
        [doi for assessment in assessments for doi in assessment["dois"]],
        doi_to_pmid,
    )

    return PrefillData(
        form_type=form_type,
        panel_id=panel_id,
        hgnc_id=f"HGNC:{hgnc_id}",
        rating=PREFILL_RATINGS[rating].upper(),
        moi=moi,
        mode_of_pathogenicity=None,
        publications=";".join(publications),
        phenotypes=";".join(phenotypes),
        comments=prefill_comments(form_type, associations) if rated else "",
    )


def entity_meets_green(entity: dict[str, Any]) -> bool:
    """Check if a single disease entity satisfies (A OR B OR C) AND D AND E on its own.

    PanelApp GREEN status requires one entity to meet the conjunction by itself —
    not pieces stitched across entities.
    """
    a = bool(get_entity_criterion(entity, "criterion_A").get("result", False))
    b = bool(get_entity_criterion(entity, "criterion_B").get("result", False))
    c = bool(get_entity_criterion(entity, "criterion_C").get("result", False))
    d = bool(get_entity_criterion(entity, "criterion_D").get("result", False))
    e = bool(get_entity_criterion(entity, "criterion_E").get("result", False))

    return (a or b or c) and d and e


def calculate_association_rating(entity: dict[str, Any]) -> int:
    """Rate one association (disease entity) on its own PanelApp criteria.

    - 3 (GREEN) if it satisfies (A OR B OR C) AND D AND E
    - 2 (AMBER) if not GREEN but it has more than one independent family
    - 1 (RED) otherwise
    """
    if entity_meets_green(entity):
        return 3
    if (entity["independent_family_count"] or 0) > 1:
        return 2
    return 1


def calculate_gene_rating(associations: list[dict[str, Any]]) -> int:
    """The gene's rating: the highest rating of its associations, RED when it has none.

    Returns:
        Confidence level: 3 (GREEN), 2 (AMBER), or 1 (RED)
    """
    return max((calculate_association_rating(a) for a in associations), default=1)


# PanelApp's rating names as confidence levels
RATING_CONFIDENCE: dict[str, int] = {"RED": 1, "AMBER": 2, "GREEN": 3}


def panelapp_confidence_to_color(confidence: int | None) -> str:
    """Convert PanelApp confidence level to color name.

    Args:
        confidence: PanelApp confidence level integer

    Returns:
        Color name (Grey, Red, Amber, Green)
    """
    if confidence is None:
        return "Grey"

    mapping = {
        0: "Grey",  # Not in panel / no evidence
        1: "Red",  # Limited evidence
        2: "Amber",  # Moderate evidence
        3: "Green",  # High evidence
    }
    return mapping.get(confidence, "Grey")
