"""Post-generation QA — checks a rendered DOCX against its source DocumentAST."""

from .docx_qa import DocxQAReport, QACheck, run_docx_qa

__all__ = ["DocxQAReport", "QACheck", "run_docx_qa"]
