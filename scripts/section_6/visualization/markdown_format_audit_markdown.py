# Section 6 visualization helper used by the notebook and report.
from ..runtime import np

def format_audit_markdown(payload: dict) -> str:
    audit = payload["meta"]["audit"]
    bullet_lines = "\n".join(f"- {item}" for item in audit["bottlenecks"])
    dense_share = 100.0 * audit["baseline_dense_params"] / audit["baseline_params"]
    return (
        "### VII.1. Diagnostic de depart\n\n"
        f"- Baseline courante : `{audit['baseline_layout']}`.\n"
        f"- Parametres entrainables de la baseline : `{audit['baseline_params']}`.\n"
        f"- Parametres concentres dans la tete dense : `{audit['baseline_dense_params']}` ({dense_share:.1f}% du total).\n"
        f"{bullet_lines}"
    )
