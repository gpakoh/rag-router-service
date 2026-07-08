import io
import logging
from pathlib import Path

from app.exceptions import ParserError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".docx", ".pdf"}


def parse_single_file_sync(filename: str, content: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ParserError(f"Unsupported file extension: '{ext}'", filename=filename)

    try:
        if ext == ".txt":
            text = _parse_txt(content)
        elif ext == ".md":
            text = _parse_md(content)
        elif ext == ".docx":
            text = _parse_docx(content)
        elif ext == ".pdf":
            text = _parse_pdf(content)
        else:
            raise ParserError(f"Unsupported file extension: '{ext}'", filename=filename)
    except ParserError:
        raise
    except Exception as e:
        raise ParserError(
            f"Failed to parse '{filename}': {e}", filename=filename
        ) from e

    if not text.strip():
        raise ParserError(f"Empty extracted text from '{filename}'", filename=filename)

    return text.strip()


def _parse_txt(content: bytes) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("utf-8", errors="replace")


def _parse_md(content: bytes) -> str:
    import markdown
    from bs4 import BeautifulSoup

    html = markdown.markdown(content.decode("utf-8", errors="replace"))
    return BeautifulSoup(html, "html.parser").get_text()


def _parse_docx(content: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs)


def _parse_pdf(content: bytes) -> str:
    import pdfplumber

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)
