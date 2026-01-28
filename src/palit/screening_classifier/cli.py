#!/usr/bin/env python3
"""Screening classifier subcommand registration."""

import typer

from palit.screening_classifier import evaluate, extract_pmids, prepare_data, train

app = typer.Typer(help="Screening classifier training and evaluation commands")

# Register subcommands
app.command("train")(train.main)
app.command("evaluate")(evaluate.main)
app.command("prepare-data")(prepare_data.main)
app.command("extract-pmids")(extract_pmids.main)


if __name__ == "__main__":
    app()
