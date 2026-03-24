from ..runtime import np

def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(str(item) for item in row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])
