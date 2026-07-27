"""
Document loaders: turn an uploaded file into plain text.

Kept intentionally simple -- one function per file type. Add more
`load_*` functions here as you support more formats (docx, html, etc).
"""
from pathlib import Path

from pypdf import PdfReader


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)
    return "\n".join(pages)


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in (".txt", ".md"):
        return load_txt(path)
    raise ValueError(f"Unsupported file type: {suffix}")
