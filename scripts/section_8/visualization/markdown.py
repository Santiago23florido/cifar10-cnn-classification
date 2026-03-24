# Section 8 formatting helper used by the notebook summaries.
def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(item) for item in row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body])


def format_audit_markdown(payload: dict) -> str:
    lines = ["### IV.1. Diagnostic", ""]
    for item in payload["meta"]["audit"]:
        lines.append(f"- {item}")
    return "\n".join(lines)


def format_protocol_markdown(payload: dict) -> str:
    protocol = payload["meta"]["protocol"]
    rows = [
        ["Reference model", protocol["reference_model"]],
        ["Reference seed", protocol["reference_seed"]],
        ["Reference epochs", protocol["reference_epochs"]],
        ["Selected layers", ", ".join(protocol["selected_layers"])],
        ["Evolution layers", ", ".join(protocol["evolution_layers"])],
        ["Selected epochs", ", ".join(str(epoch) for epoch in protocol["selected_epochs"])],
        ["Primary class", protocol["selected_classes"]["primary"]],
        ["Secondary class", protocol["selected_classes"]["secondary"]],
        ["Static normalization", protocol["visual_normalization"]["static_panels"]],
        ["Evolution normalization", protocol["visual_normalization"]["epoch_evolution"]],
    ]
    return markdown_table(["Parameter", "Value"], rows)


def format_section8_summary_markdown(payload: dict) -> str:
    selected = payload["selected_images"]
    first_diag = payload["first_layer"]["diagnostics"]
    rows = []
    for row in payload["summary_table"]:
        rows.append([row["analysis"], row["layers"], row["epochs"], row["displayed_channels"], row["observation"]])
    lines = [
        "### IV.3. Images et observations retenues",
        "",
        f"- Image principale : index `{selected['primary']['index']}`, classe `{selected['primary']['true_class_name']}`, confiance `{selected['primary']['confidence']:.4f}`.",
        f"- Image secondaire : index `{selected['secondary']['index']}`, classe `{selected['secondary']['true_class_name']}`, confiance `{selected['secondary']['confidence']:.4f}`.",
        f"- Canaux de premiere couche les plus lies au gradient : `{first_diag['top_edge_channels']}`.",
        f"- Canaux de premiere couche les plus lies au contraste chromatique : `{first_diag['top_color_channels']}`.",
        "",
        markdown_table(["Analysis", "Layers", "Epochs", "Displayed channels", "Observation"], rows),
    ]
    return "\n".join(lines)