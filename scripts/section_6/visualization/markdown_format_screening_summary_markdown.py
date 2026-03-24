from ..runtime import np

from .markdown__format_comparison_table import _format_comparison_table

def format_screening_summary_markdown(payload: dict) -> str:
    shortlisted = payload["screening"]["shortlisted_refined"]
    shortlisted_names = [item["name"] for item in payload["architecture_catalog"] if item["slug"] in shortlisted]
    body = _format_comparison_table(payload["screening"]["summary"])
    return body + "\n\n" + f"Architectures raffinees retenues pour confirmation : {', '.join(shortlisted_names)}."
