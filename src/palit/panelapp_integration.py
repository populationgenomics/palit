#!/usr/bin/env python3
"""Shared utilities for PanelApp criteria evaluation and integration."""

from dataclasses import dataclass
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


def inheritance_details_items(details: str) -> list[str]:
    """The "; "-separated items of an association's inheritance_details; none for ""."""
    return details.split("; ") if details else []


MondoMatch = Literal["panelapp_gencc", "exact", "broader"]


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


@dataclass
class PrefillData:
    """Prefill form data for PanelApp integration."""

    form_type: str  # "add" or "review"
    panel_id: int
    hgnc_id: str  # "HGNC:8607" format for PanelApp
    rating: str  # "GREEN", "AMBER", "RED"
    moi: str  # Full PanelApp MoI string
    mode_of_pathogenicity: str | None
    publications: str  # semicolon-separated identifiers (PMIDs where available, DOIs otherwise)
    phenotypes: str  # semicolon-separated "label, MONDO:id" entries or bare disease names
    comments: str  # one plain-text section per association


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


def prefill_phenotype(association: PrefillAssociation) -> str:
    """The association's entry in the prefill's phenotype field.

    A MONDO term gives ``label, MONDO:id``: its own label for a reused GenCC term
    and an exact match, the proposed disease name for a broader match. A reused
    GenCC term that MONDO obsoleted with exactly one replacement gives the
    replacement instead; with no or several replacements it stays as submitted.
    An association without a MONDO term gives its proposed disease name alone.
    """
    mondo = association.mondo
    if mondo is None:
        name: str = association.assessment["proposed_disease_name"]
        return name
    if mondo.match == "broader":
        return f"{association.assessment['proposed_disease_name']}, {mondo.mondo_id}"
    if mondo.obsolete and len(mondo.replaced_by) == 1:
        replacement = mondo.replaced_by[0]
        return f"{replacement.label}, {replacement.mondo_id}"
    return f"{mondo.label}, {mondo.mondo_id}"


def prefill_comment_section(association: PrefillAssociation) -> str:
    """The association's section of the prefill comment: a heading line, then its summary.

    The heading names the disease as the phenotype field does, flags a broader
    MONDO term with its label, and gives the MoI and the rating computed from
    this run's papers.
    """
    assessment = association.assessment
    disease = prefill_phenotype(association)
    if association.mondo is not None and association.mondo.match == "broader":
        disease += f" (broader MONDO term: {association.mondo.label})"
    moi = assessment["inheritance_mode"].replace("_", " ")
    rating = panelapp_confidence_to_color(calculate_association_rating(assessment)).upper()
    return f"{disease} | {moi} | {rating} in this corpus\n{assessment['summary']}"


def prepare_prefill_data(
    hgnc_id: int,
    associations: list[PrefillAssociation],
    form_type: str,
    panel_id: int,
    doi_to_pmid: dict[str, int | None],
) -> PrefillData:
    """One PanelApp prefill for the gene, as the union over its associations.

    The rating is the top association rating and the MoI is aggregated over all
    associations (see derive_panelapp_moi). Phenotypes, publications and comment
    sections follow the order of *associations* (the report's order); repeated
    phenotypes and publications are listed once.

    Args:
        hgnc_id: HGNC ID (integer) of the gene
        associations: The gene's associations, in report order
        form_type: "add" or "review"
        panel_id: Target panel ID
        doi_to_pmid: PMID (None when the paper has none) of every DOI the associations cite

    Returns:
        PrefillData object ready for form rendering
    """
    assessments = [a.assessment for a in associations]
    rating_str = panelapp_confidence_to_color(calculate_gene_rating(assessments)).upper()

    dois = dict.fromkeys(doi for assessment in assessments for doi in assessment["dois"])
    publications = ";".join(
        doi if (pmid := doi_to_pmid[doi]) is None else str(pmid) for doi in dois
    )

    phenotypes = ";".join(dict.fromkeys(prefill_phenotype(a) for a in associations))
    comments = "\n\n".join(prefill_comment_section(a) for a in associations)

    return PrefillData(
        form_type=form_type,
        panel_id=panel_id,
        hgnc_id=f"HGNC:{hgnc_id}",
        rating=rating_str,
        moi=derive_panelapp_moi(assessments),
        mode_of_pathogenicity=None,
        publications=publications,
        phenotypes=phenotypes,
        comments=comments,
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
