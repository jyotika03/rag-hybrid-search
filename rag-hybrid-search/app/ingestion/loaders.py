"""Multi-format document loader.

Accepts markdown, text, HTML and PDF. Normalizes to clean plaintext and
emits structured sections carrying metadata (source file, section heading,
page number). Raw bytes are preserved by the caller so re-indexing never
requires re-uploading.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


@dataclass
class Section:
    text: str
    source_document: str
    section_heading: str | None = None
    page_number: int | None = None


@dataclass
class LoadedDocument:
    source_document: str
    sections: list[Section] = field(default_factory=list)

    @property
    def plaintext(self) -> str:
        return "\n\n".join(s.text for s in self.sections)


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_markdown(path: str, name: str) -> LoadedDocument:
    raw = open(path, encoding="utf-8", errors="ignore").read()
    sections: list[Section] = []
    current_heading = None
    buf: list[str] = []

    def flush():
        if buf:
            sections.append(
                Section(_clean("\n".join(buf)), name, current_heading)
            )

    for line in raw.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush()
            buf = []
            current_heading = m.group(2).strip()
        else:
            buf.append(line)
    flush()
    if not sections:
        sections = [Section(_clean(raw), name)]
    return LoadedDocument(name, sections)


def _load_text(path: str, name: str) -> LoadedDocument:
    raw = open(path, encoding="utf-8", errors="ignore").read()
    return LoadedDocument(name, [Section(_clean(raw), name)])


def _load_html(path: str, name: str) -> LoadedDocument:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(open(path, encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    sections: list[Section] = []
    current_heading = None
    buf: list[str] = []

    def flush():
        if buf:
            txt = _clean(" ".join(buf))
            if txt:
                sections.append(Section(txt, name, current_heading))

    for el in soup.find_all(True):
        if el.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            flush()
            buf = []
            current_heading = el.get_text(strip=True)
        elif el.name in {"p", "li", "td", "pre"}:
            buf.append(el.get_text(" ", strip=True))
    flush()
    if not sections:
        sections = [Section(_clean(soup.get_text(" ")), name)]
    return LoadedDocument(name, sections)


def _load_pdf(path: str, name: str) -> LoadedDocument:
    from pypdf import PdfReader

    reader = PdfReader(path)
    sections = []
    for i, page in enumerate(reader.pages):
        txt = _clean(page.extract_text() or "")
        if txt:
            sections.append(Section(txt, name, page_number=i + 1))
    if not sections:
        sections = [Section("", name)]
    return LoadedDocument(name, sections)


LOADERS = {
    ".md": _load_markdown,
    ".markdown": _load_markdown,
    ".txt": _load_text,
    ".html": _load_html,
    ".htm": _load_html,
    ".pdf": _load_pdf,
}


def load_document(path: str) -> LoadedDocument:
    name = os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()
    loader = LOADERS.get(ext)
    if not loader:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader(path, name)


def load_directory(directory: str) -> list[LoadedDocument]:
    docs = []
    for root, _, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() in LOADERS:
                docs.append(load_document(os.path.join(root, f)))
    return docs
