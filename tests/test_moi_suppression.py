"""Tests for MoI-change suppression in report generation.

Covers ``decompose_moi`` and ``apply_moi_suppression`` — both the Rule 1
weak-expansion gate and the Rule 2 Incidentalome "already-recorded" suppression,
including the three April regression genes the curator dismissed as noise.
"""

import pytest

from palit.generate_report import apply_moi_suppression
from palit.panelapp_client import AllPanelsData
from palit.panelapp_integration import INCIDENTALOME_PANEL_ID, decompose_moi

MENDELIOME_PANEL_ID = 137


def _panels(
    gene_panel_mois: dict[int, dict[int, str]], panel_names: dict[int, str]
) -> AllPanelsData:
    """Build AllPanelsData, deriving panel membership from the per-panel MoI map."""
    gene_to_panels = {hgnc: set(panels) for hgnc, panels in gene_panel_mois.items()}
    return AllPanelsData(
        gene_to_panels=gene_to_panels,
        panel_names=panel_names,
        gene_panel_mois=gene_panel_mois,
    )


def test_decompose_moi_combined() -> None:
    assert decompose_moi("Monoallelic_and_biallelic") == frozenset({"Monoallelic", "Biallelic"})


@pytest.mark.parametrize("mode", ["Monoallelic", "Biallelic", "X-linked", "Mitochondrial", "Other"])
def test_decompose_moi_simple_is_identity(mode: str) -> None:
    assert decompose_moi(mode) == frozenset({mode})


def test_tnni3_narrowing_suppressed() -> None:
    # Incidentalome gene recorded as "both"; aggregate evidence narrows to Biallelic,
    # which is already encompassed by the existing combined enum. Classified a
    # contradiction, yet should be suppressed via decomposition of the existing side.
    panels = _panels(
        gene_panel_mois={
            11947: {
                INCIDENTALOME_PANEL_ID: "Monoallelic_and_biallelic",
                111: "Monoallelic_and_biallelic",
            }
        },
        panel_names={INCIDENTALOME_PANEL_ID: "Incidentalome", 111: "Hypertrophic cardiomyopathy"},
    )
    reason = apply_moi_suppression(
        status="contradiction",
        existing_moi="Monoallelic_and_biallelic",
        new_moi="Biallelic",
        hgnc_id=11947,
        moi_family_counts={},
        all_panels_data=panels,
    )
    assert "Biallelic already on" in reason


def test_vhl_expansion_suppressed_names_rare_panel() -> None:
    # Monoallelic Incidentalome gene; biallelic erythrocytosis is already on Red cell
    # disorders, so the mono+bi aggregate introduces nothing novel.
    panels = _panels(
        gene_panel_mois={12687: {INCIDENTALOME_PANEL_ID: "Monoallelic", 3366: "Biallelic"}},
        panel_names={INCIDENTALOME_PANEL_ID: "Incidentalome", 3366: "Red cell disorders"},
    )
    reason = apply_moi_suppression(
        status="expansion",
        existing_moi="Monoallelic",
        new_moi="Monoallelic_and_biallelic",
        hgnc_id=12687,
        moi_family_counts={"Biallelic": 5},  # well-evidenced, so Rule 1 does not fire
        all_panels_data=panels,
    )
    assert "Biallelic already on Red cell disorders" in reason


def test_sqstm1_mendeliome_gene_not_suppressed() -> None:
    # Not on the Incidentalome -> Rule 2 does not apply; monoallelic is well-evidenced
    # -> Rule 1 does not fire. The flag stays for manual panel-placement triage.
    panels = _panels(
        gene_panel_mois={
            11280: {MENDELIOME_PANEL_ID: "Biallelic", 24: "Monoallelic", 25: "Monoallelic"}
        },
        panel_names={
            MENDELIOME_PANEL_ID: "Mendeliome",
            24: "Early-onset Dementia",
            25: "Motor Neurone Disease",
        },
    )
    reason = apply_moi_suppression(
        status="expansion",
        existing_moi="Biallelic",
        new_moi="Monoallelic_and_biallelic",
        hgnc_id=11280,
        moi_family_counts={"Monoallelic": 5},
        all_panels_data=panels,
    )
    assert reason == ""


def test_novel_mode_on_incidentalome_not_suppressed() -> None:
    # On the Incidentalome, but X-linked is recorded nowhere for this gene: genuinely
    # new inheritance information, so it stays highlighted.
    panels = _panels(
        gene_panel_mois={999: {INCIDENTALOME_PANEL_ID: "Monoallelic"}},
        panel_names={INCIDENTALOME_PANEL_ID: "Incidentalome"},
    )
    reason = apply_moi_suppression(
        status="contradiction",
        existing_moi="Monoallelic",
        new_moi="X-linked",
        hgnc_id=999,
        moi_family_counts={},
        all_panels_data=panels,
    )
    assert reason == ""


def test_rule1_weak_expansion_takes_precedence() -> None:
    # Both rules would suppress (biallelic recorded elsewhere AND too few families).
    # Rule 1 runs first, so its evidence-strength reason wins over the attribution.
    panels = _panels(
        gene_panel_mois={777: {INCIDENTALOME_PANEL_ID: "Monoallelic", 3366: "Biallelic"}},
        panel_names={INCIDENTALOME_PANEL_ID: "Incidentalome", 3366: "Red cell disorders"},
    )
    reason = apply_moi_suppression(
        status="expansion",
        existing_moi="Monoallelic",
        new_moi="Monoallelic_and_biallelic",
        hgnc_id=777,
        moi_family_counts={"Biallelic": 1},
        all_panels_data=panels,
    )
    assert "threshold" in reason
    assert "Red cell disorders" not in reason
