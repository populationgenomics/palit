#!/usr/bin/env python3
"""Normalize extracted variants from papers.

Huge kudos to the Evidence Aggregator team (https://github.com/microsoft/healthfutures-evagg),
which this code is based on.
"""

import gzip
import json
import logging
import os
import re
import sqlite3
import urllib.parse as urlparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import httpx
import tenacity
import typer
from defusedxml import ElementTree
from pydantic import BaseModel, field_validator, model_validator
from rich.progress import BarColumn, SpinnerColumn, TaskProgressColumn, TextColumn

from palit.hgnc import HgncResolver
from palit.progress import LoggingProgress as Progress

logger = logging.getLogger(__name__)

app = typer.Typer(help="Normalize extracted variants from papers")


# Protein single-letter to three-letter amino acid code mapping
_PROTEIN_LETTERS_1TO3 = {
    "A": "Ala",
    "C": "Cys",
    "D": "Asp",
    "E": "Glu",
    "F": "Phe",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "K": "Lys",
    "L": "Leu",
    "M": "Met",
    "N": "Asn",
    "P": "Pro",
    "Q": "Gln",
    "R": "Arg",
    "S": "Ser",
    "T": "Thr",
    "V": "Val",
    "W": "Trp",
    "Y": "Tyr",
}

# Chromosome → NC_ accession mapping per genome build.
_CHR_TO_NC: dict[str, dict[str, str]] = {
    "GRCh37": {
        "chr1": "NC_000001.10",
        "chr2": "NC_000002.11",
        "chr3": "NC_000003.11",
        "chr4": "NC_000004.11",
        "chr5": "NC_000005.9",
        "chr6": "NC_000006.11",
        "chr7": "NC_000007.13",
        "chr8": "NC_000008.10",
        "chr9": "NC_000009.11",
        "chr10": "NC_000010.10",
        "chr11": "NC_000011.9",
        "chr12": "NC_000012.11",
        "chr13": "NC_000013.10",
        "chr14": "NC_000014.8",
        "chr15": "NC_000015.9",
        "chr16": "NC_000016.9",
        "chr17": "NC_000017.10",
        "chr18": "NC_000018.9",
        "chr19": "NC_000019.9",
        "chr20": "NC_000020.10",
        "chr21": "NC_000021.8",
        "chr22": "NC_000022.10",
        "chrX": "NC_000023.10",
        "chrY": "NC_000024.9",
    },
    "GRCh38": {
        "chr1": "NC_000001.11",
        "chr2": "NC_000002.12",
        "chr3": "NC_000003.12",
        "chr4": "NC_000004.12",
        "chr5": "NC_000005.10",
        "chr6": "NC_000006.12",
        "chr7": "NC_000007.14",
        "chr8": "NC_000008.11",
        "chr9": "NC_000009.12",
        "chr10": "NC_000010.11",
        "chr11": "NC_000011.10",
        "chr12": "NC_000012.12",
        "chr13": "NC_000013.11",
        "chr14": "NC_000014.9",
        "chr15": "NC_000015.10",
        "chr16": "NC_000016.10",
        "chr17": "NC_000017.11",
        "chr18": "NC_000018.10",
        "chr19": "NC_000019.10",
        "chr20": "NC_000020.11",
        "chr21": "NC_000021.9",
        "chr22": "NC_000022.11",
        "chrX": "NC_000023.11",
        "chrY": "NC_000024.10",
    },
}

_BUILD_ALIASES: dict[str, str] = {"hg19": "GRCh37", "hg38": "GRCh38"}


def _chr_to_nc_accession(chr_refseq: str, genome_build: str | None) -> str:
    """Resolve a chromosome designation (e.g. 'chr7') to an NC_ accession using genome build."""
    if not genome_build:
        raise ValueError(
            f"Chromosome-style refseq '{chr_refseq}' requires a genome build, but none was provided"
        )
    normalized_build = _BUILD_ALIASES.get(genome_build, genome_build)
    build_map = _CHR_TO_NC.get(normalized_build)
    if not build_map:
        raise ValueError(
            f"Unknown genome build '{genome_build}' for chromosome refseq '{chr_refseq}'"
        )
    # Normalize chrX/chrY/chr1 etc. — strip anything after the chromosome number/letter
    chr_key = re.match(r"(chr[\dXY]+)", chr_refseq, re.IGNORECASE)
    if not chr_key:
        raise ValueError(f"Cannot parse chromosome from refseq '{chr_refseq}'")
    nc = build_map.get(chr_key.group(1).lower().replace("chr", "chr"))
    # Try case-insensitive lookup
    if not nc:
        for k, v in build_map.items():
            if k.lower() == chr_key.group(1).lower():
                nc = v
                break
    if not nc:
        raise ValueError(f"Unknown chromosome '{chr_key.group(1)}' for build '{normalized_build}'")
    return nc


@dataclass(frozen=True)
class HGVSVariant:
    """A representation of a genetic variant."""

    hgvs_desc: str
    refseq: str

    def __str__(self) -> str:
        """Obtain a string representation of the variant."""
        return f"{self.refseq}:{self.hgvs_desc}"


@dataclass(frozen=True)
class PseudoVCFVariant:
    p_vcf: str  # 8-42437272-C-A
    hgvs_c: str  # NM_001257180.1:c.1240G>T
    hgvs_p: str  # NP_001244109.1:p.Glu414Ter


class VariantNormalizer:
    def __init__(self) -> None:
        self._web_client = _WebClient()
        self._ncbi_client = _NcbiClient(web_client=self._web_client)
        self._refseq_client = _RefSeqClient(
            web_client=self._web_client, ncbi_client=self._ncbi_client
        )
        self._mutalyzer_client = _MutalyzerClient(web_client=self._web_client)

    def pseudo_vcf(self, hgvs_variant: HGVSVariant) -> list[PseudoVCFVariant]:
        """Attempts to normalize a variant to pseudo-VCF (17-50198002-C-A) format,
        which gnomAD uses as variant IDs.

        This isn't always unqiuely possible: e.g. for HGVS p. variants, the back-translation
        from protein variant to coding variant can return multiple possibilities.

        Raises an exception in case the conversion fails.
        """

        variant_descriptions = [str(hgvs_variant)]

        def remove_parens(variant_description: str) -> str:
            return variant_description.replace("(", "").replace(")", "")

        # For protein variants, attempt back-translation.
        if hgvs_variant.hgvs_desc.startswith("p."):
            # Attempt back-translation.
            back_translated = self._mutalyzer_client.back_translate(variant_descriptions[0])
            # Remove uncertainty parens: NM_001257180.2:c.(1240G>T) -> NM_001257180.2:c.1240G>T,
            # as otherwise the variant validator look-up below will fail.
            variant_descriptions = [remove_parens(bt) for bt in back_translated]

        # Call the VariantValidator API to get the genomic coordinates.
        p_vcfs = []
        for variant_description in variant_descriptions:
            encoded = urlparse.quote(variant_description)
            url = f"https://rest.variantvalidator.org/VariantValidator/variantvalidator/GRCh38/{encoded}/mane_select"

            response = self._web_client.get(url, content_type="json")

            lookup_key = variant_description
            if hgvs_variant.hgvs_desc.startswith("m."):
                lookup_key = "mitochondrial_variant_1"
            entry = response.get(lookup_key)

            # VariantValidator returns results keyed by transcript, not by the submitted
            # variant description. Try known fallback patterns.
            if entry is None and "(" in hgvs_variant.refseq:
                # Genomic-wrapped format: NC_000007.14(NM_003592.3):c.483+1G>A
                transcript = hgvs_variant.refseq.split("(")[1].rstrip(")")
                transcript_key = f"{transcript}:{hgvs_variant.hgvs_desc}"
                entry = response.get(transcript_key)
            if entry is None:
                # Genomic g. variants: response is keyed by the transcript variant
                # (e.g. NM_xxx:c.xxx). Find the first NM_ key.
                for key in response:
                    if key.startswith("NM_"):
                        entry = response[key]
                        break
            if entry is None:
                entry = {}

            vcf = entry.get("primary_assembly_loci", {}).get("grch38", {}).get("vcf")

            if not vcf:
                raise ValueError(f"Failed to find genomic coordinates for {hgvs_variant}")

            p_vcf = f"{vcf['chr']}-{vcf['pos']}-{vcf['ref']}-{vcf['alt']}"
            hgvs_c = entry["hgvs_transcript_variant"]
            hgvs_p = remove_parens(entry["hgvs_predicted_protein_consequence"]["tlr"])

            p_vcfs.append(PseudoVCFVariant(p_vcf, hgvs_c, hgvs_p))

        return p_vcfs

    def hgvs(self, variant_str: str, gene_symbol: str, genome_build: str | None) -> HGVSVariant:
        """Normalize a variant string to HGVS format.

        This method takes a variant string (which may be in various formats) and attempts
        to parse and normalize it into a standardized HGVSVariant representation. It handles
        both dbSNP rsIDs and HGVS-like variant descriptions.

        Args:
            variant_str: The variant string to normalize (e.g., "c.123A>G", "rs12345", "p.Arg123Ter")
            gene_symbol: The gene symbol associated with the variant (required for protein/coding variants)
            genome_build: The genome build for genomic coordinates (e.g., "GRCh38", only required for genomic coordinates)

        Returns:
            HGVSVariant: A normalized variant representation containing the HGVS description and RefSeq accession

        Raises:
            ValueError: If the variant string cannot be parsed or if required parameters are missing
        """

        # If the variant_str contains a dbsnp rsid, parse it and return the variant.
        if matched := re.match(r".*?:?(rs\d+).*?", variant_str):
            try:
                variant = self._parse_rsid(matched.group(1))
                if not variant:
                    raise ValueError(
                        f"Unable to create variant from {variant_str} and {gene_symbol}"
                    )
                return variant
            except ValueError as e:
                raise ValueError(
                    f"Unable to create variant from {variant_str} and {gene_symbol}"
                ) from e

        # Otherwise, assume we're working with an hgvs-like variant.

        # Remove all the spaces from the variant string.
        variant_str = variant_str.replace(" ", "")

        # Occassionally gene_symbol is embedded in variant_str, if it is, we'll have to extract it.
        # This is generally either of the form gene_symbol:variant or gene_symbol(variant). Sometimes,
        # gene_symbol is prefixed with a 'g' (e.g., pmid:33117677).
        variant_str = re.sub(f"g?{gene_symbol}:", "", variant_str)
        search_result = re.search(r"g?" + gene_symbol + r"\((.*?)\)", variant_str)
        if search_result:
            variant_str = search_result.group(1)

        # Split out the refseq if it's present.
        if variant_str.find(":") >= 0:
            refseq, variant_str = variant_str.split(":", 1)
            refseq = refseq.strip()
            variant_str = variant_str.strip()
        else:
            refseq = None
            variant_str = variant_str.strip()

        # If the variant string looks nothing like a variant description, give up.
        if not re.search(r"[A-Za-z]", variant_str):
            raise ValueError(f"Variant string '{variant_str}' appears unparsable.")

        # If the refseq looks like a chromosome designation, resolve to the actual NC_ accession.
        is_chromosomal = refseq is not None and "chr" in refseq
        if is_chromosomal:
            assert refseq is not None
            refseq = _chr_to_nc_accession(refseq, genome_build)
        # Otherwise, it should begin with NM_, NP_, or NC_, otherwise we'll ignore it.
        elif refseq and not re.match(r"(NM_|NP_|NC_)", refseq):
            logger.debug(f"Ignoring potentially invalid refseq: {refseq}")
            refseq = None

        # Remove any parentheses and brackets.
        variant_str = variant_str.replace("(", "").replace(")", "")
        variant_str = variant_str.replace("[", "").replace("]", "")

        # To handle variants where splitting failed, remove everything before the first semicolon.
        variant_str = variant_str.split(";")[0]

        # To handle previxes that weren't removed, remove everything up through the last colon.
        variant_str = variant_str.split(":")[-1]

        # Occassionally, protein level descriptions do not include the p. prefix, add it if it's missing.
        # This will only currently handle fairly simple protein level descriptions.
        if re.search(r"^[A-Za-z]+\d+[A-Za-z]+$", variant_str):
            variant_str = "p." + variant_str

        # Occassionally, coding/genomic level descriptions do not include the c./g. prefix.
        # Use g. for chromosomal refseqs, c. for transcript refseqs.
        bare_prefix = "g." if is_chromosomal else "c."
        if re.search(r"^\d+[ACGT]>[ACGT]$", variant_str):
            variant_str = bare_prefix + variant_str
        if re.search(r"^\d+(_\d+)?del[ACGT]*$", variant_str):
            variant_str = bare_prefix + variant_str
        if re.search(r"^\d+ins[ACGT]*$", variant_str):
            variant_str = bare_prefix + variant_str

        # Single-letter protein level descriptions should use * for a stop codon, not X or stop.
        variant_str = re.sub(r"(p\.[A-Z]\d+)X", r"\1*", variant_str)
        variant_str = re.sub(r"(p\.[A-Z]\d+)stop", r"\1*", variant_str)

        # Fix c. descriptions that are erroneously written as c.{ref}{pos}{alt} instead of c.{pos}{ref}>{alt}.
        variant_str = re.sub(r"c\.([ACTG])(\d+)([A-Z]+)", r"c.\2\1>\3", variant_str)

        # Fix three-letter p. descriptions that don't follow the capitalization convention.
        # For now, only handle reference AAs and single missense alternate AAs.
        if "del" not in variant_str:
            if match := re.match(
                r"p\.([A-Za-z][a-z]{2})(\d+)([A-Za-z][a-z]{2})*(.*?)$", variant_str
            ):
                ref_aa, pos, alt_aa, extra = match.groups()
                variant_str = (
                    f"p.{ref_aa.capitalize()}{pos}{alt_aa.capitalize() if alt_aa else ''}{extra}"
                )

        # Frameshift should be designated with fs, not frameshift
        variant_str = variant_str.replace("frameshift", "fs")

        # If there's a hypen that's not surrounded by numbers, remove it.
        variant_str = re.sub(r"(?<!\d)-(?!\d)", "", variant_str)

        # Remove everything after the first occurrence of "fs" if it occurs,
        # HGVS nomenclature gets variable in these cases in practice.
        if "fs" in variant_str:
            variant_str = variant_str.split("fs")[0] + "fs"

        try:
            return self._parse_hgvs(variant_str, gene_symbol, refseq)
        except ValueError as e:
            raise ValueError(
                f"Unable to create variant from {variant_str} and {gene_symbol}"
            ) from e

    def _parse_hgvs(
        self, text_desc: str, gene_symbol: str | None, refseq: str | None = None
    ) -> HGVSVariant:
        """Attempt to parse a variant based on description and an optional gene symbol and optional refseq.

        `gene_symbol` is required for protein (p.) and coding (c.) variants, but not mitochondrial (m.) or genomic (g.)
        variants.

        `refseq` is required for genomic (g.) variants. For other variant types, if a `refseq` is not provided, it will
        be predicted based on the variant description gene symbol.

        Raises a ValueError if the above requirements are not met.
        """
        if (text_desc.startswith("p.") or text_desc.startswith("c.")) and not gene_symbol:
            raise ValueError(f"Gene symbol required for protein and coding variants: {text_desc}")

        refseq = self._clean_refseq(refseq, text_desc, gene_symbol)
        return self._normalize_and_create(text_desc, refseq)

    def _clean_refseq(self, refseq: str | None, text_desc: str, gene_symbol: str | None) -> str:
        # If no refseq is provided, we'll try to predict it. If one is provided, we'll make sure it's versioned
        # correctly and complete.
        if not refseq:
            refseq = self._predict_refseq(text_desc, gene_symbol)
        elif refseq.find(".") < 0:
            refseq_replacement = self._refseq_client.accession_autocomplete(refseq)
            if refseq_replacement:
                refseq = refseq_replacement

        if not refseq:
            raise ValueError(f"No RefSeq provided or predicted for {text_desc, gene_symbol}")

        # If the variant is intronic, the refseq should be either a transcript with an included genomic reference, or
        # should be a standalone genomic reference.
        #   see https://hgvs-nomenclature.org/stable/background/refseq/
        if text_desc.find("+") >= 0 or text_desc.find("-") >= 0:
            # Intronic sequence variant, ensure that we've got an NG_ or NC_ refseq to start
            if refseq.startswith("NM_"):
                # Find the associated genomic reference sequence.
                logger.debug(
                    f"Intronic variant without genomic reference, attempting to fix: {text_desc} {gene_symbol}"
                )
                if gene_symbol:
                    chrom_refseq = self._refseq_client.genomic_accession_for_symbol(gene_symbol)
                    refseq = f"{chrom_refseq}({refseq})" if chrom_refseq else refseq

        return refseq

    def _predict_refseq(self, text_desc: str, gene_symbol: str | None) -> str | None:
        """Predict the RefSeq for a variant based on its description and gene symbol."""
        if text_desc.startswith("p.") and gene_symbol:
            protein_accession = self._refseq_client.protein_accession_for_symbol(gene_symbol)
            if transcript_accession := self._refseq_client.transcript_accession_for_symbol(
                gene_symbol
            ):
                return f"{transcript_accession}({protein_accession})"
            return protein_accession
        elif text_desc.startswith("c.") and gene_symbol:
            return self._refseq_client.transcript_accession_for_symbol(gene_symbol)
        elif text_desc.startswith("m."):
            MITO_REFSEQ = "NC_012920.1"
            return MITO_REFSEQ
        elif text_desc.startswith("g."):
            raise ValueError(
                f"Genomic (g. prefixed) variants must have a RefSeq. None was provided for {text_desc}"
            )
        else:
            logger.warning(
                "Unable to predict refseq for variant with unknown HGVS "
                f"type: {text_desc} with gene symbol {gene_symbol}."
            )
            return None

    def _normalize_and_create(self, text_desc: str, refseq: str) -> HGVSVariant:
        normalized = self._mutalyzer_client.normalize(f"{refseq}:{text_desc}")

        # Normalize the variant description.
        if "normalized_description" in normalized:
            normalized_hgvs = normalized["normalized_description"]
            refseq = normalized_hgvs.split(":")[0]
            new_text_desc = normalized_hgvs.split(":")[1]
            if new_text_desc.find("(") >= 0 and new_text_desc.find(")") >= 0:
                text_desc = new_text_desc.replace("(", "").replace(")", "")
            else:
                text_desc = new_text_desc
            # If this is a protein variant, we only want the NP_ portion of the refseq.
            if text_desc.startswith("p."):
                match = re.search(r"NP_\d+\.\d+", refseq)
                refseq = match.group(0) if match else refseq

        # Construct and return the resulting variant.
        return HGVSVariant(hgvs_desc=text_desc, refseq=refseq)

    def _parse_rsid(self, rsid: str) -> HGVSVariant:
        """Parse a variant based on an rsid."""
        hgvs_lookup = self._hgvs_from_rsid(rsid)
        full_hgvs = None
        gene_symbol = None

        if rsid in hgvs_lookup:
            if "hgvs_c" in hgvs_lookup[rsid]:
                full_hgvs = hgvs_lookup[rsid]["hgvs_c"]
            elif "hgvs_p" in hgvs_lookup[rsid]:
                full_hgvs = hgvs_lookup[rsid]["hgvs_p"]
            elif "hgvs_g" in hgvs_lookup[rsid]:
                full_hgvs = hgvs_lookup[rsid]["hgvs_g"]
            gene_symbol = hgvs_lookup[rsid].get("gene")

        if not full_hgvs:
            raise ValueError(f"Could not find HGVS for info rsid {rsid}")

        refseq = full_hgvs.split(":")[0]
        text_desc = full_hgvs.split(":")[1]

        return self._parse_hgvs(text_desc, gene_symbol, refseq)

    def _hgvs_from_rsid(self, *rsids: str) -> dict[str, dict[str, str]]:
        # Provided rsids should be numeric strings prefixed with `rs`.
        if not rsids or not all(rsid.startswith("rs") and rsid[2:].isnumeric() for rsid in rsids):
            raise ValueError(
                "Invalid rsids list - must provide 'rs' followed by a string of numeric characters."
            )

        uids = {rsid[2:] for rsid in rsids}
        try:
            root = self._ncbi_client._efetch(
                db="snp", id=",".join(uids), retmode="xml", rettype="xml"
            )
        except httpx.HTTPStatusError as e:
            logger.warning(f"Unexpected error fetching HGVS data for rsids {','.join(uids)}: {e}")
            return {}

        return {"rs" + uid: self._extract_hgvs_from_xml(root, uid) for uid in uids}

    def _extract_hgvs_from_xml(self, root: Any, uid: str) -> dict[str, str]:
        if root is None:
            return {}
        ns = "{https://www.ncbi.nlm.nih.gov/SNP/docsum}"
        # Find the first DOCSUM node under a DocumentSummary with the given rsid in the document hierarchy.
        node = next(iter(root.findall(f"./{ns}DocumentSummary[@uid='{uid}']/{ns}DOCSUM")), None)
        if node is None or not node.text:
            return {}

        # Extract all key/value pairs from node text of the form 'key=value|key=value|...' into a dict.
        props = {
            k: v
            for k, v in (kvp.split("=") for kvp in (node.text or "").split("|") if "=" in kvp)
            if k and v
        }
        # Extract all values from the HGVS property of the form 'HGVS=value1,value2...'.
        hgvs = props.get("HGVS", "").split(",")
        gene = props.get("GENE", "").split(":")[0]

        # Return a dict with the first occurrence of each value that starts with 'NP_' (hgvs_p), 'NM_' (hgvs_c), or 'NC_'
        # (genomic reference sequences / non-coding variants).
        types = {
            "hgvs_p": lambda x: x.startswith("NP_"),
            "hgvs_c": lambda x: x.startswith("NM_"),
            "hgvs_g": lambda x: x.startswith("NC_"),
        }
        ret_dict = {
            k: next(filter(match, hgvs)) for k, match in types.items() if (any(map(match, hgvs)))
        }
        ret_dict["gene"] = gene
        return ret_dict


class _WebClientSettings(BaseModel):
    model_config = {"extra": "forbid"}
    max_retries: int = 1  # Only retry once for transient failures
    retry_backoff: float = 0.5  # indicates progression of 0.5, 1, 2, 4, 8, etc. seconds
    retry_codes: list[int] = [429, 500, 502, 503, 504]  # rate-limit exceeded, server errors
    content_type: str = "text"
    timeout: float = 10.0  # seconds - fail fast on timeouts

    @field_validator("content_type")
    @classmethod
    def _validate_content_type(cls, value: str) -> str:
        CONTENT_TYPES = ["text", "json", "xml"]
        if value not in CONTENT_TYPES:
            raise ValueError(
                f"Web content type must be one of {'/'.join(CONTENT_TYPES)}, got '{value}'"
            )
        return value


class _WebClient:
    """A web content client that uses httpx + tenacity."""

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self._settings = _WebClientSettings(**settings) if settings else _WebClientSettings()
        self._client = httpx.Client(follow_redirects=True)
        self._get_content_with_retry = tenacity.retry(
            stop=tenacity.stop_after_attempt(1 + self._settings.max_retries),
            wait=tenacity.wait_exponential(multiplier=self._settings.retry_backoff),
            retry=tenacity.retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
            reraise=True,
        )(self._get_content)

    def _raise_for_status(self, url: str, code: int) -> None:
        """Raise an exception if the status code is not 2xx."""
        if 400 <= code < 600:
            raise httpx.HTTPStatusError(
                f"Request failed with status code {code} for {url}",
                request=httpx.Request("GET", url),
                response=httpx.Response(code),
            )

    def _transform_content(self, text: str, content_type: str | None) -> Any:
        """Get the content from the response based on the provided content type."""
        content_type = content_type or self._settings.content_type
        if content_type == "text":
            return text
        elif content_type == "json":
            return json.loads(text) if text else {}
        elif content_type == "xml":
            return ElementTree.fromstring(text) if text else None
        else:
            raise ValueError(f"Invalid content type: {content_type}")

    def _get_content(self, url: str, data: dict[str, Any] | None = None) -> tuple[int, str]:
        """GET (or POST) the text content at the provided URL.

        Raises httpx.HTTPStatusError for retriable status codes so tenacity can retry them.
        Non-retriable codes are returned normally for the caller to handle.
        """
        if data is not None:
            response = self._client.post(url, json=data, timeout=self._settings.timeout)
        else:
            response = self._client.get(url, timeout=self._settings.timeout)
        if response.status_code in self._settings.retry_codes:
            raise httpx.HTTPStatusError(
                f"Request failed with status code {response.status_code} for {url}",
                request=response.request,
                response=response,
            )
        return response.status_code, response.text

    def get(
        self,
        url: str,
        data: dict[str, Any] | None = None,
        content_type: str | None = None,
        url_extra: str | None = None,
    ) -> Any:
        """GET (or POST) the content at the provided URL."""
        full_url = f"{url}{url_extra or ''}"
        code, content = self._get_content_with_retry(full_url, data)
        self._raise_for_status(full_url, code)
        return self._transform_content(content, content_type)


class _NcbiClientSettings(BaseModel):
    model_config = {"extra": "forbid"}
    api_key: str | None = None
    email: str = "panelapp-support@mcri.edu.au"

    def get_key_string(self) -> str | None:
        key_string = ""
        if self.email:
            key_string += f"&email={urlparse.quote(self.email)}"
        if self.api_key:
            key_string += f"&api_key={self.api_key}"
        return key_string if key_string else None

    @model_validator(mode="before")
    @classmethod
    def _validate_settings(cls, values: dict[str, Any]) -> dict[str, Any]:
        if values.get("api_key") and not values.get("email"):
            raise ValueError("If NCBI_EUTILS_API_KEY is specified NCBI_EUTILS_EMAIL is required.")
        return values


class _NcbiClient:
    EUTILS_HOST = "https://eutils.ncbi.nlm.nih.gov"
    EUTILS_SEARCH_SITE = "/entrez/eutils/esearch.fcgi"
    EUTILS_FETCH_SITE = "/entrez/eutils/efetch.fcgi"
    EUTILS_SEARCH_URL = EUTILS_SEARCH_SITE + "?db={db}&term={term}&sort={sort}&tool=biopython"
    EUTILS_FETCH_URL = (
        EUTILS_FETCH_SITE + "?db={db}&id={id}&retmode={retmode}&rettype={rettype}&tool=biopython"
    )

    def __init__(self, web_client: _WebClient, settings: dict[str, str] | None = None) -> None:
        self._config = _NcbiClientSettings(**settings) if settings else _NcbiClientSettings()
        self._web_client = web_client

    def _esearch(self, db: str, term: str, sort: str, **extra_params: dict[str, Any]) -> Any:
        key_string = self._config.get_key_string()
        url = self.EUTILS_SEARCH_URL.format(db=db, term=term, sort=sort)
        url += "".join([f"&{k}={v}" for k, v in extra_params.items()])
        return self._web_client.get(
            f"{self.EUTILS_HOST}{url}", content_type="xml", url_extra=key_string
        )

    def _efetch(
        self, db: str, id: str, retmode: str | None = None, rettype: str | None = None
    ) -> Any:
        key_string = self._config.get_key_string()
        url = self.EUTILS_FETCH_URL.format(db=db, id=id, retmode=retmode, rettype=rettype)
        return self._web_client.get(
            f"{self.EUTILS_HOST}{url}", content_type=retmode, url_extra=key_string
        )


class _RefSeqClient:
    """Determine RefSeq 'Mane Select' and 'RefSeq Select' accessions for genes using the NCBI RefSeq database."""

    _NCBI_REFSEQ_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/vertebrate_mammalian/Homo_sapiens/all_assembly_versions/GCF_000001405.39_GRCh38.p13/GCF_000001405.39_GRCh38.p13_genomic.gff.gz"
    _RAW_FILENAME = "GCF_000001405.39_GRCh38.p13_genomic.gff.gz"
    _PROCESSED_FILEPATH = "refseq_processed.json"

    _DEFAULT_REFERENCE_DIR = "data/"

    def __init__(
        self,
        web_client: _WebClient,
        ncbi_client: _NcbiClient,
        reference_dir: str = _DEFAULT_REFERENCE_DIR,
    ) -> None:
        self._web_client = web_client
        self._ncbi_client = ncbi_client
        self._reference_dir = reference_dir
        self._ref: dict[str, Any]

        # Download the reference file if necessary.
        if not os.path.exists(self._reference_dir):
            logging.info(f"Creating reference directory at {self._reference_dir}")
            os.makedirs(self._reference_dir)

        resource_path = os.path.join(self._reference_dir, self._PROCESSED_FILEPATH)

        if not os.path.exists(resource_path):
            self._ref = self._process_raw_resource()
            json.dump(self._ref, open(resource_path, "w"), indent=4)
        else:
            self._ref = json.load(open(resource_path))

    def transcript_accession_for_symbol(self, symbol: str) -> str | None:
        """Get the RefSeq transcript accession for a gene symbol."""
        return cast(str | None, self._ref.get(symbol, {}).get("RNA", None))

    def protein_accession_for_symbol(self, symbol: str) -> str | None:
        """Get the RefSeq protein accession for a gene symbol."""
        return cast(str | None, self._ref.get(symbol, {}).get("Protein", None))

    def genomic_accession_for_symbol(self, symbol: str) -> str | None:
        """Get the RefSeq genomic accession for a gene symbol."""
        return cast(str | None, self._ref.get(symbol, {}).get("Genomic", None))

    def accession_autocomplete(self, accession: str) -> str | None:
        """Get the latest RefSeq version for a versionless accession."""
        if accession.find(".") >= 0:
            logger.info(f"Accession '{accession}' is already versioned. Nothing to do.")
            return accession

        result = self._ncbi_client._efetch(
            db="nuccore", id=accession, retmode="text", rettype="acc"
        )
        result = cast(str, result).strip()

        if result.startswith("Error") is False:
            return result
        return None

    def _download_binary_reference(self, url: str, target: str) -> None:
        response = httpx.get(url, timeout=60, follow_redirects=True)
        response.raise_for_status()

        with open(target, "wb") as f:
            f.write(response.content)

    def _process_raw_resource(self) -> dict[str, Any]:
        raw_target_filepath = os.path.join(self._reference_dir, self._RAW_FILENAME)

        if not os.path.exists(raw_target_filepath):
            logger.info(f"Downloading reference file to {raw_target_filepath}")
            print(f"Downloading reference file to {raw_target_filepath}")

            self._download_binary_reference(self._NCBI_REFSEQ_URL, raw_target_filepath)

        protein_lines = []
        transcript_lines = []

        refseqs: dict[str, Any] = {}

        logger.info("Processing raw reference file.")
        with gzip.open(raw_target_filepath, "rt") as f:
            for line in f:
                if not re.search(r"(MANE|RefSeq) Select", line):
                    continue
                if re.search(r"BestRefSeq\s+CDS", line):
                    protein_lines.append(line)
                elif re.search(r"BestRefSeq\s+mRNA", line):
                    transcript_lines.append(line)

        os.remove(raw_target_filepath)

        for line in protein_lines:
            tokens = line.split("\t")
            assert len(tokens) == 9, "Wrong number of tokens in line"
            attributes = {kv.split("=")[0]: kv.split("=")[1] for kv in tokens[8].split(";")}
            assert "gene" in attributes, "gene not found in attributes"
            assert "tag" in attributes and "Select" in attributes["tag"], "Not tagged as select"
            assert "protein_id" in attributes, "protein_id not found in attributes"
            assert "Dbxref" in attributes, "Dbxref not found in attributes"
            xref = {kv.split(":")[0]: kv.split(":")[1] for kv in attributes["Dbxref"].split(",")}
            assert "GeneID" in xref, "GeneID not found in DBxref"

            if attributes["gene"] in refseqs:
                if refseqs[attributes["gene"]]["MANE"]:
                    logger.warning(f"{attributes['gene']} already has a MANE protein")
                    continue

            refseqs[attributes["gene"]] = {
                "Protein": attributes["protein_id"],
                "Genomic": tokens[0],
                "Symbol": attributes["gene"],
                "GeneID": xref["GeneID"],
                "MANE": "MANE Select" in attributes["tag"],
            }

        for line in transcript_lines:
            tokens = line.split("\t")
            assert len(tokens) == 9, "Wrong number of tokens in line"
            attributes = {kv.split("=")[0]: kv.split("=")[1] for kv in tokens[8].split(";")}
            assert "gene" in attributes, "gene not found in attributes"
            assert "tag" in attributes and "Select" in attributes["tag"], "Not tagged as Select"
            assert "transcript_id" in attributes, "transcript_id not found in attributes"

            if attributes["gene"] not in refseqs:
                # This is uncommon, so log a warning.
                logger.warning(f"{attributes['gene']} not found in proteins")
                continue

            if "RNA" in refseqs[attributes["gene"]]:
                # This is relatively common, because a gene can have multiple transcripts.
                logger.debug(f"{attributes['gene']} already has an RNA")
                continue

            if "MANE Select" in attributes["tag"] and not refseqs[attributes["gene"]]["MANE"]:
                # No observed instances of this in the current reference data, but it's possible, so log a warning.
                logger.warning(f"{attributes['gene']} has a non-MANE protein, but a MANE RNA.")

            refseqs[attributes["gene"]]["RNA"] = attributes["transcript_id"].strip()

        logger.info(f"Processed {len(refseqs)} RefSeq entries.")

        return refseqs


class _MutalyzerClient:
    _web_client: _WebClient

    def __init__(self, web_client: _WebClient) -> None:
        self._web_client = web_client

    def _normalize_frame_shift(self, hgvs: str) -> dict[str, Any]:
        """Normalize a frame shift variant using a custom approach."""
        # fs variants are of any of the following forms
        # - p.XNNNfs
        # - p.XNNNYfs
        # - p.XNNNYfs*
        # - p.XNNNYfs*7
        #
        # X and Y can be a one-letter or three-letter amino acid, N is a number, and * is a stop codon,
        # which can alternatively be expressed as Ter.
        #
        # p.(XNNNfs) and p.XNNNfs are both acceptable, the normalized form should retain parentheses.
        #
        # The normalized representation should be p.XNNNfs where X is the three letter amino acid code.

        logger.debug(f"Normalizing frame shift variant {hgvs}")

        refseq, hgvs_desc = hgvs.split(":")

        # Drop anything past NNN and replace with fs.
        hgvs_desc = re.sub(r"(\(?)([A-Za-z]+[0-9]+)[A-Za-z0-9\*]+(\)?)", r"\1\2fs\3", hgvs_desc)

        # Now replace the single letter code with the three letter code, if that's what was used.
        if matched := re.match(r"(p.\(?)([A-Z])([0-9]+fs\)?)", hgvs_desc):
            hgvs_desc = (
                matched.group(1) + _PROTEIN_LETTERS_1TO3[matched.group(2)] + matched.group(3)
            )

        return {"normalized_description": f"{refseq}:{hgvs_desc}"}

    def normalize(self, hgvs: str) -> dict[str, Any]:
        """Normalize an HGVS description using Mutalyzer.

        hgvs: The HGVS description to normalize, e.g., NM_000551.3:c.1582G>A
        """
        # Mutalyzer doesn't currently support normalizing frame shift variants, so we have to take our own approach
        # here.
        if hgvs.split(":")[1].find("fs") != -1:
            return self._normalize_frame_shift(hgvs)

        # Preprocess the hgvs string to avoid some common issues.
        # Remove whitespace and unicode whitespace
        hgvs = hgvs.replace(" ", "").replace("\u2009", "")

        # Replace forward slashes with commas, these are occasionally used to designate multiple alternate alleles,
        # but they are not valid in a URL and urrlib.parse.quote doesn't solve the issue.
        hgvs = hgvs.replace("/", ",")

        # Encode the hgvs string for use in a URL.
        encoded = urlparse.quote(hgvs)

        # 422 = unprocessable entity (syntactically or biologically invalid description).
        # 500 = Mutalyzer bug for specific variants (e.g., NP_000099.2:p.R316X); treat as unresolvable.
        url = f"https://mutalyzer.nl/api/normalize/{encoded}"

        try:
            response = self._web_client.get(url, content_type="json")
        except httpx.HTTPStatusError as e:
            logger.debug(f"{url} returned an error: {e}")
            if e.response.status_code in (422, 500):
                return {"error_message": f"Mutalyzer error ({e.response.status_code})"}
            raise

        if "errors" in response or ("custom" in response and "errors" in response["custom"]):
            error_dict = response.get("errors") or response["custom"]["errors"]
            error_message = error_dict[0].get("code", "Unknown error")

            logger.debug(
                f"Unable to normalize variant. Mutalyzer returned an error for {hgvs}: {error_message}"
            )
            return {"error_message": error_message}

        # Only return a subset of the fields in the response.
        response_dict = {}
        if "normalized_description" in response:
            response_dict["normalized_description"] = response["normalized_description"]
        if "protein" in response and "description" in response["protein"]:
            response_dict["protein"] = {"description": response["protein"]["description"]}
        if "equivalent_descriptions" in response:
            response_dict["equivalent_descriptions"] = response["equivalent_descriptions"]

        return response_dict

    def validate(self, hgvs: str) -> tuple[bool, str | None]:
        """Validate an HGVS description using Mutalyzer."""
        # Mutalyzer doesn't currently support normalizing frame shift variants, so we can't validate them.
        # TODO, consider tweaking to be a stop gain and normalizing that.
        if ":" not in hgvs:
            return (False, "Invalid HGVS description")

        if hgvs.split(":")[1].find("fs") != -1:
            logger.debug(f"Skipping validation of frame shift variant {hgvs}")
            return (False, "Frameshift validation not supported")

        normalization_result = self.normalize(hgvs)
        error_message = normalization_result.pop("error_message", None)
        return (bool(normalization_result), error_message)

    def back_translate(self, hgvsp: str) -> list[str]:
        """Back translate a protein variant description to a coding variant description using Mutalyzer.

        hgvsp: The protein variant description to back translate. Must conform to HGVS nomenclature.
        """
        # Mutalyzer doesn't currently support normalizing frame shift variants, so we can't back-translate them.
        if hgvsp.split(":")[1].find("fs") != -1:
            raise ValueError(f"Back-translation of frameshift variants not supported: {hgvsp}")

        encoded = urlparse.quote(hgvsp)
        url = f"https://mutalyzer.nl/api/back_translate/{encoded}"

        return cast(list[str], self._web_client.get(url, content_type="json"))


# Code below is just for standalone testing.


@dataclass
class _VariantNormalizationResult:
    """Result of normalizing a single variant."""

    hgnc_id: int
    original: str
    normalized: str | None
    p_vcf: PseudoVCFVariant | None
    error: str | None


@dataclass
class _PaperVariants:
    """Variants extracted from a single paper."""

    doi: str
    genome_build: str | None
    gene_variants: list[tuple[int, list[str]]] = field(default_factory=list)  # (hgnc_id, variants)


@dataclass
class _PaperVariantStats:
    """Statistics for variants from a single paper."""

    doi: str
    total_variants: int
    successful_normalizations: int
    failed_normalizations: int


def _load_extracted_variants(db_path: Path) -> list[_PaperVariants]:
    """Load extracted variants from database grouped by DOI.

    Args:
        db_path: Path to SQLite database

    Returns:
        List of PaperVariants objects
    """
    logger.info(f"Loading extracted variants from {db_path}...")

    paper_variants_list: list[_PaperVariants] = []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all papers with evidence extraction
        cursor.execute("""
            SELECT doi, evidence_extraction_json
            FROM papers
            WHERE evidence_extraction_json IS NOT NULL
            ORDER BY doi DESC
        """)

        for row in cursor.fetchall():
            doi = row["doi"]

            try:
                extraction_data = json.loads(row["evidence_extraction_json"])
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse evidence extraction for DOI {doi}")
                continue

            # Get genome build (may be None or "unknown")
            genome_build = extraction_data.get("genome_build")
            if genome_build == "unknown":
                genome_build = None

            # Extract variants from each gene evaluation
            gene_evaluations = extraction_data.get("gene_evaluations", [])
            gene_variants_list = []

            for eval_data in gene_evaluations:
                hgnc_id = eval_data.get("hgnc_id")
                if hgnc_id is None:
                    continue
                variants = eval_data.get("variants", [])
                variant_strs = [variant["variant"] for variant in variants]
                gene_variants_list.append((hgnc_id, variant_strs))

            if gene_variants_list:
                paper_variants = _PaperVariants(
                    doi=doi, genome_build=genome_build, gene_variants=gene_variants_list
                )
                paper_variants_list.append(paper_variants)

    logger.info(f"Loaded variants from {len(paper_variants_list)} papers")
    return paper_variants_list


def _process_variants(
    paper_variants_list: list[_PaperVariants],
    hgnc_resolver: HgncResolver,
) -> tuple[dict[str, list[_VariantNormalizationResult]], list[_PaperVariantStats]]:
    """Process and normalize variants.

    Args:
        paper_variants_list: List of PaperVariants objects
        hgnc_resolver: HGNC resolver for gene symbol lookup

    Returns:
        Tuple of (normalization results by DOI, statistics per paper)
    """

    variant_normalizer = VariantNormalizer()

    results_by_doi: dict[str, list[_VariantNormalizationResult]] = {}
    stats_per_paper: list[_PaperVariantStats] = []

    # Calculate total variants to process
    total_papers = len(paper_variants_list)
    total_variants_to_process = sum(
        sum(len(variants) for _hgnc_id, variants in paper.gene_variants)
        for paper in paper_variants_list
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("({task.completed}/{task.total})"),
    ) as progress:
        # Create progress bars
        paper_task = progress.add_task("[cyan]Processing papers...", total=total_papers)
        variant_task = progress.add_task(
            "[yellow]Normalizing variants...", total=total_variants_to_process
        )

        for paper in paper_variants_list:
            paper_results: list[_VariantNormalizationResult] = []
            total_variants = 0
            successful = 0
            failed = 0

            for hgnc_id, variants in paper.gene_variants:
                gene_symbol = hgnc_resolver.get_symbol(hgnc_id)
                for variant_text in variants:
                    total_variants += 1

                    # Update variant progress description
                    progress.update(
                        variant_task,
                        description=f"[yellow]DOI {paper.doi}: {variant_text[:30]}...",
                    )

                    try:
                        logger.debug(
                            f"Normalizing variant '{variant_text}' for HGNC:{hgnc_id} from DOI {paper.doi}"
                        )

                        hgvs_variant = variant_normalizer.hgvs(
                            variant_text, gene_symbol, paper.genome_build
                        )

                        p_vcfs = variant_normalizer.pseudo_vcf(hgvs_variant)

                        # Simply append all results, but typically one would want to consider non-unique
                        # translations together (e.g. taking the maximum gnomAD population frequency
                        # across all possibilities).
                        paper_results.extend(
                            _VariantNormalizationResult(
                                hgnc_id=hgnc_id,
                                original=variant_text,
                                normalized=str(hgvs_variant),
                                p_vcf=p_vcf,
                                error=None,
                            )
                            for p_vcf in p_vcfs
                        )

                        successful += 1
                    except Exception as e:
                        paper_results.append(
                            _VariantNormalizationResult(
                                hgnc_id=hgnc_id,
                                original=variant_text,
                                normalized=None,
                                p_vcf=None,
                                error=str(e),
                            )
                        )

                        failed += 1
                        logger.debug(
                            f"Failed to normalize variant '{variant_text}' for HGNC:{hgnc_id} from DOI {paper.doi}: {e}"
                        )

                    # Update variant progress
                    progress.update(variant_task, advance=1)

            if paper_results:
                results_by_doi[paper.doi] = paper_results
                stats_per_paper.append(
                    _PaperVariantStats(
                        doi=paper.doi,
                        total_variants=total_variants,
                        successful_normalizations=successful,
                        failed_normalizations=failed,
                    )
                )

            # Update paper progress
            progress.update(paper_task, advance=1)

    return results_by_doi, stats_per_paper


def _print_results(
    results_by_doi: dict[str, list[_VariantNormalizationResult]],
    stats_per_paper: list[_PaperVariantStats],
) -> None:
    """Print normalized variants and statistics to stdout.

    Args:
        results_by_doi: Normalization results grouped by DOI
        stats_per_paper: Statistics for each paper
    """
    # Print results by DOI
    for doi in sorted(results_by_doi.keys(), reverse=True):
        print(f"\n## DOI {doi}")
        print("-" * 40)

        for result in results_by_doi[doi]:
            if result.normalized is not None:
                assert result.p_vcf is not None
                print(
                    f"{result.original}: {result.normalized} ({result.p_vcf.p_vcf}, {result.p_vcf.hgvs_c}, {result.p_vcf.hgvs_p})"
                )
            else:
                print(f"{result.original}: ERROR - {result.error}")

    # Calculate and print statistics
    print("\n" + "=" * 50)
    print("STATISTICS")
    print("=" * 50)

    total_papers = len(stats_per_paper)
    total_variants = sum(s.total_variants for s in stats_per_paper)
    total_successful = sum(s.successful_normalizations for s in stats_per_paper)
    total_failed = sum(s.failed_normalizations for s in stats_per_paper)

    if total_papers > 0:
        avg_variants_per_paper = total_variants / total_papers
        error_rate = (total_failed / total_variants * 100) if total_variants > 0 else 0.0

        print(f"Total variants: {total_variants}")
        print(f"Total papers: {total_papers}")
        print(f"Average variants per paper: {avg_variants_per_paper:.1f}")
        print(
            f"Successfully normalized: {total_successful} ({total_successful / total_variants * 100:.1f}%)"
        )
        print(f"Failed normalizations: {total_failed} ({error_rate:.1f}%)")
    else:
        print("No variants found in database")


@app.command("run")
def normalize(
    db_path: Path = typer.Option(
        default=Path("data/db.sqlite"), help="Path to SQLite database with extracted variants"
    ),
    use_test_data: bool = typer.Option(
        default=False, help="Use hardcoded test data instead of loading from database"
    ),
) -> None:
    """Normalize all extracted variants from papers."""

    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        raise typer.Exit(1)

    hgnc_resolver = HgncResolver.from_file()

    # Load variants from database or use test data
    if use_test_data:
        logger.info("Using hardcoded test data")

        def _resolve_id(symbol: str) -> int:
            entry = hgnc_resolver.resolve(symbol)
            if entry is None:
                raise ValueError(f"Test gene {symbol} not found in HGNC data")
            return entry.hgnc_id

        paper_variants_list = [
            _PaperVariants(
                doi="10.1002/ajmg.a.63982",
                genome_build="GRCh38",
                gene_variants=[
                    (
                        _resolve_id("SLC20A2"),
                        [
                            "c.1240G>T",
                            "p.(Glu414*)",
                            "c.1220C>A",
                            "p.(Ser407*)",
                            "c.1696A>T",
                            "p.(Ile566Phe)",
                            "c.1703C>T",
                            "p.(Pro568Leu)",
                        ],
                    ),
                    (_resolve_id("PDGFB"), ["c.571C>T", "p.(Arg191*)", "c.418C>T", "p.(Gln140*)"]),
                    (_resolve_id("MYORG"), ["c.1727G>A", "p.(Arg576His)"]),
                    (_resolve_id("MAP3K6"), ["c.322G>A", "p.(Asp108Asn)"]),
                    (_resolve_id("GLA"), ["c.394G>C", "p.(Gly132Arg)"]),
                    (_resolve_id("MT-TL1"), ["m.3243A>C"]),
                ],
            )
        ]
    else:
        logger.info("Loading variants from database")
        paper_variants_list = _load_extracted_variants(db_path)

    if not paper_variants_list:
        logger.warning("No variants found in database")
        print("No variants found in database")
        return

    # Process and normalize variants
    results_by_doi, stats_per_paper = _process_variants(paper_variants_list, hgnc_resolver)

    # Print results
    _print_results(results_by_doi, stats_per_paper)

    logger.info("✅ Variant normalization complete")


@app.command("debug-variant")
def debug_variant(
    variant: str = typer.Argument(..., help="Variant text (e.g., 'c.483+1G>A')"),
    gene: str = typer.Argument(..., help="Gene symbol (e.g., 'CUL1')"),
    genome_build: str | None = typer.Option(default=None, help="Genome build (e.g., 'GRCh38')"),
) -> None:
    """Debug normalization pipeline for a single variant.

    Example: uv run normalize-variants debug-variant "c.483+1G>A" CUL1
    """
    logging.basicConfig(level=logging.DEBUG)

    normalizer = VariantNormalizer()

    print(f"Input: variant={variant!r}, gene={gene!r}, genome_build={genome_build!r}")
    print("-" * 60)

    try:
        hgvs_variant = normalizer.hgvs(variant, gene, genome_build)
        print(f"✅ HGVS: {hgvs_variant}")
    except Exception as e:
        print(f"❌ HGVS normalization failed: {e}")
        raise

    try:
        p_vcfs = normalizer.pseudo_vcf(hgvs_variant)
        print(f"✅ Pseudo-VCF ({len(p_vcfs)} result(s)):")
        for p_vcf in p_vcfs:
            print(f"   {p_vcf.p_vcf}  |  {p_vcf.hgvs_c}  |  {p_vcf.hgvs_p}")
    except Exception as e:
        print(f"❌ Pseudo-VCF conversion failed: {e}")
        raise


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
