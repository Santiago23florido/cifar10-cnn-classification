from .markdown_markdown_table import markdown_table

def format_protocol_markdown(protocol: dict) -> str:
    optimizer_params = protocol["optimizer_hyperparameters"]
    rows = [
        ["Sections de reference", "Section II pour les donnees et la standardisation, Section III pour l'architecture"],
        ["Sous-ensembles", f"{protocol['subset_sizes']['train']} train / {protocol['subset_sizes']['validation']} validation / {protocol['subset_sizes']['test']} test"],
        ["Graine de partition", protocol["split_seed"]],
        ["Graines d'entrainement", ", ".join(str(seed) for seed in protocol["training_seeds"])],
        ["Comparaison batch size", f"{protocol['batch_study_epochs']} epochs fixes, batch sizes {', '.join(str(value) for value in protocol['batch_sizes'])}"],
        ["Comparaison optimiseurs", f"{protocol['optimizer_study_epochs']} epochs fixes, batch size {protocol['optimizer_batch_size']}"],
        ["SGD", f"lr={optimizer_params['SGD']['learning_rate']}, momentum={optimizer_params['SGD']['momentum']}"],
        ["SGD+Momentum", f"lr={optimizer_params['SGD+Momentum']['learning_rate']}, momentum={optimizer_params['SGD+Momentum']['momentum']}"],
        ["Adam", f"lr={optimizer_params['Adam']['learning_rate']}, beta_1={optimizer_params['Adam']['beta_1']}, beta_2={optimizer_params['Adam']['beta_2']}, epsilon={optimizer_params['Adam']['epsilon']}"],
        ["Convention de loss", protocol['loss_reduction']],
        ["Temps/step", protocol['step_time_definition']],
        ["Temps/epoch", protocol['epoch_time_definition']],
    ]
    return markdown_table(["Parametre", "Valeur"], rows)
