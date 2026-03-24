# Notebook imports for Section 6 visual summaries.
from .figures import (
    draw_confusion,
    save_all_section6_figures,
    save_final_architecture,
    save_final_confusion_matrices,
    save_final_curves,
)
from .markdown import (
    format_architecture_catalog_markdown,
    format_audit_markdown,
    format_confirmation_summary_markdown,
    format_dimension_summary_markdown,
    format_hyperparameter_summary_markdown,
    format_protocol_markdown,
    format_screening_summary_markdown,
    format_selected_model_markdown,
    markdown_table,
)
from .paths import DEFAULT_FIGURE_DIR, load_image_paths
from .payload import coerce_payload
