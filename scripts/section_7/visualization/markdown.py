# Section 7 visualization helper used by the notebook and report.
def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(str(item) for item in row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])


def format_audit_markdown(payload: dict) -> str:
    lines = ["### VIII.1. Diagnostic", ""]
    for item in payload["meta"]["audit"]:
        lines.append(f"- {item}")
    return "\n".join(lines)


def format_protocol_markdown(payload: dict) -> str:
    protocol = payload["meta"]["protocol"]
    optimizer = protocol["optimizer"]
    rows = [
        ["Dataset", protocol["dataset"]],
        ["Subset sizes", f"{protocol['subset_sizes']['train']} train / {protocol['subset_sizes']['validation']} val / {protocol['subset_sizes']['test']} test"],
        ["Preprocessing", protocol["preprocessing"]],
        ["Overfitting example", protocol["example_source"]],
        ["Study seeds", ", ".join(str(seed) for seed in protocol["study_seeds"])],
        ["Study epochs", protocol["study_epochs"]],
        ["Batch size", protocol["batch_size"]],
        ["Optimizer", f"Adam(lr={optimizer['learning_rate']}, beta_1={optimizer['beta_1']}, beta_2={optimizer['beta_2']}, epsilon={optimizer['epsilon']})"],
        ["Mechanisms tested", "; ".join(protocol["mechanisms_tested"])],
        ["Mechanisms not tested", "; ".join(protocol["mechanisms_not_tested"])],
    ]
    return markdown_table(["Parameter", "Value"], rows)


def format_example_markdown(payload: dict) -> str:
    example = payload["overfitting_example"]
    indicators = example["indicators"]
    run = example["run"]
    return "\n".join([
        "### VIII.3. Exemple de sur-apprentissage reutilise depuis la Section 6",
        "",
        f"- Source : `M2`, screening de la Section 6, seed `{run['seed']}`, `{run['epochs']}` epochs.",
        f"- Minimum de validation loss : `{indicators['min_val_loss']:.4f}` a l epoch `{indicators['epoch_at_min_val_loss']}`.",
        f"- Validation loss finale : `{indicators['final_val_loss']:.4f}`.",
        f"- Augmentation apres le minimum : `{indicators['overfit_loss_increase']:.4f}`.",
        f"- Accuracy train/val finale : `{100.0 * indicators['final_train_accuracy']:.2f}% / {100.0 * indicators['final_val_accuracy']:.2f}%`.",
        f"- Ecart final train-val : `{100.0 * indicators['final_accuracy_gap']:.2f}` points.",
    ])


def format_regularization_summary_markdown(payload: dict) -> str:
    rows = []
    for row in payload["regularization_study"]["summary"]:
        rows.append([
            row["variant_name"],
            row["mechanism"],
            f"{100.0 * row['mean_final_val_accuracy']:.2f} +/- {100.0 * row['std_final_val_accuracy']:.2f}",
            f"{row['mean_min_val_loss']:.4f} +/- {row['std_min_val_loss']:.4f}",
            f"{row['mean_overfit_loss_increase']:.4f} +/- {row['std_overfit_loss_increase']:.4f}",
            f"{100.0 * row['mean_final_accuracy_gap']:.2f} +/- {100.0 * row['std_final_accuracy_gap']:.2f}",
        ])
    return "\n".join([
        "### VIII.4. Etude comparative des mecanismes de regularisation",
        "",
        markdown_table(
            [
                "Variant",
                "Mechanism",
                "Final val. acc. (%)",
                "Min val. loss",
                "Overfit increase",
                "Train-val gap (pts)",
            ],
            rows,
        ),
    ])
