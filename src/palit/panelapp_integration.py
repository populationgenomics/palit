#!/usr/bin/env python3
"""Shared utilities for PanelApp criteria evaluation and integration."""

from dataclasses import dataclass
from typing import Any, Literal

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
    phenotypes: str  # semicolon-separated "description, MONDO_ID" pairs
    comments: str  # summary text


def derive_aggregate_moi(disease_entities: list[dict[str, Any]]) -> tuple[str, str]:
    """Derive overall inheritance mode and details from disease_entities.

    PanelApp requires a single MoI per gene, but our schema captures MoI per
    disease entity. This function aggregates across entities for PanelApp compatibility.

    Args:
        disease_entities: List of disease entity dicts, each with inheritance_mode
            and inheritance_details fields

    Returns:
        Tuple of (inheritance_mode, inheritance_details)
        - If all entities have the same mode -> that mode
        - If mixed Monoallelic + Biallelic -> Monoallelic_and_biallelic
        - If mixed with X-linked/Mitochondrial -> Other
        - If all NR -> NR
    """
    modes: set[str] = set()
    details_list: list[str] = []

    for entity in disease_entities:
        mode = entity.get("inheritance_mode")
        if mode and mode != "NR":
            modes.add(mode)
        detail = entity.get("inheritance_details")
        if detail and detail.strip():
            details_list.append(detail)

    # Combine details (unique, sorted)
    combined_details = "; ".join(sorted(set(details_list))) if details_list else ""

    # Drop "Other" when more specific modes are present (e.g. a somatic
    # mosaicism entity alongside several Monoallelic entities should not
    # force the aggregate to "Other").
    specific_modes = modes - {"Other"}
    if specific_modes:
        modes = specific_modes

    # Derive mode
    if not modes:
        return "NR", combined_details

    if len(modes) == 1:
        return modes.pop(), combined_details

    # Multiple modes - check for mono+bi combination
    if modes == {"Monoallelic", "Biallelic"}:
        return "Monoallelic_and_biallelic", combined_details

    # If Monoallelic_and_biallelic is already in there with either mono or bi
    if "Monoallelic_and_biallelic" in modes:
        remaining = modes - {"Monoallelic_and_biallelic", "Monoallelic", "Biallelic"}
        if not remaining:
            return "Monoallelic_and_biallelic", combined_details

    # Mixed with X-linked, Mitochondrial, or Other
    return "Other", combined_details


def prepare_prefill_data(
    hgnc_id: int,
    associations: list[dict[str, Any]],
    form_type: str,
    panel_id: int,
    cited_papers: list[tuple[str, int | None]],
) -> PrefillData:
    """Prepare prefill form data from a gene's associations.

    Args:
        hgnc_id: HGNC ID (integer) of the gene
        associations: Each association's assessment JSON with its row's ``mondo_id`` and
            ``mondo_label`` added (both None while the association has no MONDO term),
            in report order
        form_type: "add" or "review"
        panel_id: Target panel ID
        cited_papers: List of (doi, pmid) pairs for papers cited in the assessment

    Returns:
        PrefillData object ready for form rendering
    """
    rating_str = panelapp_confidence_to_color(calculate_gene_rating(associations)).upper()

    inheritance_mode, _ = derive_aggregate_moi(associations)
    moi = ENUM_TO_PANELAPP_MOI[inheritance_mode]

    # Format publications: use PMID where available, DOI otherwise
    publications = ";".join(str(pmid) if pmid is not None else doi for doi, pmid in cited_papers)

    # Semicolon-separated "label, MONDO_ID" pairs; an association without a MONDO term
    # contributes its proposed disease name alone.
    phenotype_entries: set[str] = set()
    for association in associations:
        mondo_id = association["mondo_id"]
        if mondo_id is None:
            phenotype_entries.add(association["proposed_disease_name"])
        else:
            phenotype_entries.add(f"{association['mondo_label']}, {mondo_id}")
    phenotypes = ";".join(sorted(phenotype_entries))

    comments = "\n\n".join(association["summary"] for association in associations)

    return PrefillData(
        form_type=form_type,
        panel_id=panel_id,
        hgnc_id=f"HGNC:{hgnc_id}",
        rating=rating_str,
        moi=moi,
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
