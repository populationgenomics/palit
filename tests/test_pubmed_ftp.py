"""Tests for the NCBI FTP fetch primitives (parsing, ordering, checksums)."""

import pytest

from palit import pubmed_ftp
from palit.pubmed_ftp import RemoteFile


def test_remote_file_name_formatting() -> None:
    assert RemoteFile(26, 1335).name == "pubmed26n1335.xml.gz"
    assert RemoteFile(26, 1).name == "pubmed26n0001.xml.gz"


def test_remote_file_ordering_survives_annual_rollover() -> None:
    # A new year sorts after the old one despite the numbering reset.
    files = [RemoteFile(27, 1), RemoteFile(26, 1497), RemoteFile(26, 1335)]
    assert sorted(files) == [RemoteFile(26, 1335), RemoteFile(26, 1497), RemoteFile(27, 1)]


def test_file_regex_parses_listing_and_excludes_md5() -> None:
    listing = (
        '<a href="pubmed26n1335.xml.gz">pubmed26n1335.xml.gz</a>'
        '<a href="pubmed26n1335.xml.gz.md5">pubmed26n1335.xml.gz.md5</a>'
        '<a href="pubmed26n1336.xml.gz">pubmed26n1336.xml.gz</a>'
    )
    found = {RemoteFile(int(y), int(n)) for y, n in pubmed_ftp._FILE_RE.findall(listing)}
    assert found == {RemoteFile(26, 1335), RemoteFile(26, 1336)}


def test_parse_md5_real_format() -> None:
    text = "MD5(pubmed26n1497.xml.gz)= 5dfab57821dba99506ba2ee6c08fade9"
    assert pubmed_ftp.parse_md5(text) == "5dfab57821dba99506ba2ee6c08fade9"


def test_parse_md5_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        pubmed_ftp.parse_md5("not a checksum")


def test_read_baseline_max_parses_readme(monkeypatch: pytest.MonkeyPatch) -> None:
    readme = (
        "The complete baseline consists of files pubmed26n0001.xml "
        "through pubmed26n1334.xml.\n"
        "The first Update file to be loaded ... is pubmed26n1335.xml."
    )
    monkeypatch.setattr(pubmed_ftp, "_get_text", lambda client, url: readme)
    assert pubmed_ftp.read_baseline_max(client=None) == RemoteFile(26, 1334)
