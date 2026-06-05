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
        total = entity.get("family_count")
        independent = entity.get("independent_family_count")
        if total is None or independent is None:
            if total is not None or independent is not None:
                return False
            continue
        if not 0 <= independent <= total:
            return False
    return True


# MONDO ID to category information (abbreviation, description)
# CSS class is derived from abbrev.lower()
MONDO_CATEGORIES = {
    "MONDO:0002254": {"abbrev": "Synd", "label": "Syndromic disease"},
    "MONDO:0003778": {"abbrev": "IEI", "label": "Inborn error of immunity"},
    "MONDO:0005047": {"abbrev": "IF", "label": "Infertility disorder"},
    "MONDO:0044970": {"abbrev": "Mito", "label": "Mitochondrial disease"},
    "MONDO:0700092": {"abbrev": "NDD", "label": "Neurodevelopmental disorder"},
}

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

# Reverse mapping from PanelApp long-form to enum (1:1 from canonical)
PANELAPP_MOI_TO_ENUM = {v: k for k, v in ENUM_TO_PANELAPP_MOI.items() if v}

# Additional PanelApp long-forms that map to the same enum values
PANELAPP_MOI_TO_ENUM.update(
    {
        "MONOALLELIC, autosomal or pseudoautosomal, NOT imprinted": "Monoallelic",
        "MONOALLELIC, autosomal or pseudoautosomal, maternally imprinted (paternal allele expressed)": "Monoallelic",
        "MONOALLELIC, autosomal or pseudoautosomal, paternally imprinted (maternal allele expressed)": "Monoallelic",
        "MONOALLELIC, autosomal or pseudoautosomal, imprinted status unknown": "Monoallelic",
        "BOTH monoallelic and biallelic (but BIALLELIC mutations cause a more SEVERE disease form), autosomal or pseudoautosomal": "Monoallelic_and_biallelic",
        "X-LINKED: hemizygous mutation in males, biallelic mutations in females": "X-linked",
        "Unknown": "Other",
    }
)


def decompose_moi(moi: str) -> frozenset[str]:
    """Split an MoI enum into its constituent simple inheritance modes.

    "Monoallelic_and_biallelic" is the only combined enum; every other value
    (Monoallelic, Biallelic, X-linked, Mitochondrial, Other) is already simple
    and maps to itself. Lets us compare an aggregate MoI against per-panel MoIs
    at the mode level, since each PanelApp panel records one mode per gene.
    """
    if moi == "Monoallelic_and_biallelic":
        return frozenset({"Monoallelic", "Biallelic"})
    return frozenset({moi})


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


def count_families_by_moi(disease_entities: list[dict[str, Any]]) -> dict[str, int]:
    """Largest single-entity independent family count per inheritance mode.

    Each disease entity is an independent gene-disease association, so the
    relevant per-MoI count is the strongest single entity, not the sum. Reads
    independent_family_count; when that is null (NR) it falls back to
    patient_count (a pre-existing heuristic for the MoI-expansion highlight only).

    Monoallelic_and_biallelic is a legacy enum value; entries carrying it
    contribute their count to both Monoallelic and Biallelic.

    Args:
        disease_entities: List of disease entity dicts

    Returns:
        Dict mapping inheritance mode to the max single-entity family count
    """
    counts: dict[str, int] = {}

    def update_max(moi: str, count: int) -> None:
        if count > counts.get(moi, 0):
            counts[moi] = count

    for entity in disease_entities:
        moi = entity.get("inheritance_mode")
        if not moi or moi == "NR":
            continue

        count = entity["independent_family_count"]
        if count is None:  # NR family count — fall back to patients (highlight heuristic only)
            count = entity.get("patient_count", 0)
        if count <= 0:
            continue

        if moi == "Monoallelic_and_biallelic":
            update_max("Monoallelic", count)
            update_max("Biallelic", count)
        else:
            update_max(moi, count)

    return counts


def prepare_prefill_data(
    hgnc_id: int,
    assessment_json: dict[str, Any],
    form_type: str,
    panel_id: int,
    cited_papers: list[tuple[str, int | None]],
) -> PrefillData:
    """Prepare prefill form data from gene assessment.

    Args:
        hgnc_id: HGNC ID (integer) of the gene
        assessment_json: Assessment JSON with criteria evaluations
        form_type: "add" or "review"
        panel_id: Target panel ID
        cited_papers: List of (doi, pmid) pairs for papers cited in the assessment

    Returns:
        PrefillData object ready for form rendering
    """
    # Calculate rating from assessment
    rating = calculate_gene_rating(assessment_json)
    rating_str = panelapp_confidence_to_color(rating).upper()

    # Get MoI in PanelApp long format (derived from disease_entities)
    disease_entities = assessment_json.get("disease_entities", [])
    inheritance_mode, _ = derive_aggregate_moi(disease_entities)
    moi = ENUM_TO_PANELAPP_MOI[inheritance_mode]

    # Get mode of pathogenicity if present
    mode_of_pathogenicity = assessment_json.get("mode_of_pathogenicity")

    # Format publications: use PMID where available, DOI otherwise
    publications = ";".join(str(pmid) if pmid is not None else doi for doi, pmid in cited_papers)

    # Format phenotypes as semicolon-separated "label, MONDO_ID" pairs
    disease_entities = assessment_json["disease_entities"]
    mondo_pairs: set[str] = set()
    for entity in disease_entities:
        mondo_id = entity["mondo_id"]
        category = MONDO_CATEGORIES.get(mondo_id)
        label = category["label"] if category else entity.get("mondo_label", mondo_id)
        mondo_pairs.add(f"{label}, {mondo_id}")
    phenotypes = ";".join(sorted(mondo_pairs))

    # Get summary as comments
    comments = assessment_json.get("summary", "")

    return PrefillData(
        form_type=form_type,
        panel_id=panel_id,
        hgnc_id=f"HGNC:{hgnc_id}",
        rating=rating_str,
        moi=moi,
        mode_of_pathogenicity=mode_of_pathogenicity,
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


def meets_green_criteria(gene_eval: dict[str, Any]) -> bool:
    """Check if the gene reaches GREEN: at least one disease entity satisfies
    `(A OR B OR C) AND D AND E` on its own.
    """
    return any(entity_meets_green(entity) for entity in gene_eval.get("disease_entities", []))


def calculate_gene_rating(gene_eval: dict[str, Any]) -> int:
    """Calculate gene rating confidence level based on PanelApp criteria.

    Rating logic:
    - 3 (GREEN) if at least one disease entity satisfies (A OR B OR C) AND D AND E by itself
    - 2 (AMBER) if not GREEN but more than one independent family for any phenotype
    - 1 (RED) otherwise

    Args:
        gene_eval: Gene evaluation dictionary with disease_entities each carrying their
            own evidence_assessments (5 criteria) and evidence_weakening_factors.

    Returns:
        Confidence level: 3 (GREEN), 2 (AMBER), or 1 (RED)
    """
    if meets_green_criteria(gene_eval):
        return 3

    disease_entities = gene_eval.get("disease_entities", [])
    if disease_entities:
        max_independent = max(
            entity["independent_family_count"] or 0 for entity in disease_entities
        )
        if max_independent > 1:
            return 2

    return 1


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
