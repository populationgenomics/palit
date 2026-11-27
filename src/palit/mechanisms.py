"""The mechanism of disease of a gene-disease-MoI association, as its stages share it.

The mechanisms are the Gene Curation Coalition's four categories: loss of function,
gain of function, dominant negative, and "not loss of function" for evidence that
loss of function is not the mechanism. Each is judged on its own, so an association
can hold several, each with a confidence term (Established or Inconclusive) and a
statement of the evidence. An association that holds none has an unknown mechanism.

read-mechanisms reads each paper's PDF for the mechanism evidence it holds on the
associations that cite it. assess-mechanisms judges each association's mechanisms
from the readings of its papers. The report shows them, and offers them as
structured fields of the association's assessment form.
"""

import json
from dataclasses import dataclass
from typing import Any

# The mechanisms, in the order forms and reports list them
MECHANISM_LABELS: dict[str, str] = {
    "loss_of_function": "Loss of function",
    "gain_of_function": "Gain of function",
    "dominant_negative": "Dominant negative",
    "not_loss_of_function": "Not loss of function",
}

# The confidence terms, weaker first. Inconclusive covers every mechanism that direct
# evidence supports without establishing it, which GenCC divides into Likely and Suspected.
CONFIDENCE_TERMS: tuple[str, ...] = ("Inconclusive", "Established")

# The highest term for an association whose gene-disease evidence is limited (rated RED)
LIMITED_EVIDENCE_CAP = "Inconclusive"

# The colour each confidence term is shown in. Only Established takes a colour; the
# ratings' amber and red are left to the ratings.
CONFIDENCE_COLOURS: dict[str, str] = {
    "Established": "green",
    "Inconclusive": "grey",
}


@dataclass(frozen=True)
class EvidenceItem:
    """One piece of mechanism evidence that a paper holds for an association."""

    mechanism: str  # a key of MECHANISM_LABELS
    evidence_type: str
    finding: str
    quote: str


@dataclass(frozen=True)
class AssociationReading:
    """What one paper holds on the mechanism of one association that cites it."""

    association_id: int
    authors_conclusion: str  # one sentence, or "NR"
    evidence: tuple[EvidenceItem, ...]


def readings_from_json(reading_json: str) -> dict[int, AssociationReading]:
    """A stored ``mechanism_readings.reading_json`` by association id."""
    return {
        entry["association_id"]: AssociationReading(
            association_id=entry["association_id"],
            authors_conclusion=entry["authors_conclusion"],
            evidence=tuple(EvidenceItem(**item) for item in entry["evidence"]),
        )
        for entry in json.loads(reading_json)["associations"]
    }


@dataclass(frozen=True)
class AssertedMechanism:
    """One mechanism that assess-mechanisms asserts for an association."""

    mechanism: str  # a key of MECHANISM_LABELS
    confidence: str  # one of CONFIDENCE_TERMS
    statement: str  # cites papers by the IDs the aggregation gave them
    dois: tuple[str, ...]  # the papers the statement rests on

    @property
    def label(self) -> str:
        return MECHANISM_LABELS[self.mechanism]

    @property
    def colour(self) -> str:
        """The colour of the confidence term: green or grey."""
        return CONFIDENCE_COLOURS[self.confidence]


@dataclass(frozen=True)
class MechanismAssessment:
    """An association's mechanisms, the best supported first; unknown when it has none."""

    mechanisms: tuple[AssertedMechanism, ...]
    unknown_note: str  # why the mechanism is unknown; empty unless it is

    @property
    def unknown(self) -> bool:
        return not self.mechanisms

    def confidence_of(self, mechanism: str) -> str | None:
        """The confidence term of *mechanism*, or None when it is not asserted."""
        return next((m.confidence for m in self.mechanisms if m.mechanism == mechanism), None)

    def statement_of(self, mechanism: str) -> str:
        """The statement of *mechanism*, or an empty string when it is not asserted."""
        return next((m.statement for m in self.mechanisms if m.mechanism == mechanism), "")

    def to_json(self) -> dict[str, Any]:
        return {
            "mechanisms": [
                {
                    "mechanism": m.mechanism,
                    "confidence": m.confidence,
                    "statement": m.statement,
                    "dois": list(m.dois),
                }
                for m in self.mechanisms
            ],
            "unknown_note": self.unknown_note,
        }

    def notes(self) -> str:
        """The mechanisms as a review comment carries them: a heading, then one list item
        per asserted mechanism with its confidence term and statement, or one item that
        says why the mechanism is unknown.

        A review form holds the four mechanisms as ratings only, as it holds the mode of
        inheritance, so their evidence belongs in the comment. The text reads as plain
        text and as Markdown.
        """
        items = [f"- {m.label} ({m.confidence.lower()}): {m.statement}" for m in self.mechanisms]
        if self.unknown:
            items = [f"- Unknown: {self.unknown_note}"]
        return "\n".join(["Mechanism of disease:", *items])

    @staticmethod
    def from_json(data: dict[str, Any]) -> "MechanismAssessment":
        return MechanismAssessment(
            mechanisms=tuple(
                AssertedMechanism(
                    mechanism=m["mechanism"],
                    confidence=m["confidence"],
                    statement=m["statement"],
                    dois=tuple(m["dois"]),
                )
                for m in data["mechanisms"]
            ),
            unknown_note=data["unknown_note"],
        )


GROUPING_HEADING = "Lumping and splitting:"


def review_comment(
    summary: str, grouping_statement: str | None, mechanisms: MechanismAssessment | None
) -> str:
    """The comment of an association's review, as paragraphs: its summary, its lumping
    and splitting statement once explain-grouping has written it, and the notes on its
    mechanisms of disease (see :meth:`MechanismAssessment.notes`) once assess-mechanisms
    has run for it."""
    paragraphs = [summary]
    if grouping_statement is not None:
        paragraphs.append(f"{GROUPING_HEADING} {grouping_statement}")
    if mechanisms is not None:
        paragraphs.append(mechanisms.notes())
    return "\n\n".join(paragraphs)
