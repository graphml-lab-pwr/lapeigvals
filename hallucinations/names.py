"""
Definition of names used for results presentation (e.g. paper, tables, etc.).
"""

LLM_MAP = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama3.1-8B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama3.2-3B",
    "microsoft/Phi-3.5-mini-instruct": "Phi3.5",
    "mistralai/Mistral-Nemo-Instruct-2407": "Mistral-Nemo",
    "mistralai/Mistral-Small-24B-Instruct-2501": "Mistral-Small-24B",
}

PROBE_MAP_TEX = {
    "hidden_state_last_input_token": "hidden_state_last_input_token",
    "hidden_state_last_generated_token": "hidden_state_last_generated_token",
    "attn_score": r"$\attnscore$",
    "attn_log_det": r"$\attnlogdet$",
    "attn_eigval_topk": r"$\attneig$",
    "laplacian_eigval_topk": r"$\lapeig$",
}

PROBE_MAP_PLOT = {
    "hidden_state_last_input_token_per_layer": "HiddenStateLastInputToken",
    "hidden_state_last_generated_token_per_layer": "HiddenStateLastGeneratedToken",
    "attn_log_det_per_layer": "AttnLogDet",
    "attn_eigval_topk_per_layer": "AttnEigval",
    "laplacian_eigval_topk_per_layer": "LapEigval",
    "hidden_state_last_input_token_all_layers": "HiddenStateLastInputToken (all layers)",
    "hidden_state_last_generated_token_all_layers": "HiddenStateLastGeneratedToken (all layers)",
    "attn_log_det_all_layers": "AttnLogDet (all layers)",
    "attn_eigval_topk_all_layers": "AttnEigval (all layers)",
    "laplacian_eigval_topk_all_layers": "LapEigval (all layers)",
}

MODEL_COLORS = {
    "AttnEigval": "tab:blue",
    "LapEigval": "tab:orange",
    "AttnLogDet": "tab:green",
    "AttnEigval (all layers)": "tab:blue",
    "LapEigval (all layers)": "tab:orange",
    "AttnLogDet (all layers)": "tab:green",
    "HiddenStateLastInputToken": "tab:red",
    "HiddenStateLastGeneratedToken": "tab:purple",
    "HiddenStateLastInputToken (all layers)": "tab:red",
    "HiddenStateLastGeneratedToken (all layers)": "tab:purple",
}

METRIC_MAP_TEX = {
    "auc": "AUROC",
    "average_precision": "AP",
}

CHECKMARK_MAP_TEX = {
    True: r"\checkmark",
    False: "",
}

DS_MAP = {
    "coqa": "CoQA",
    "halueval_qa": "HaluevalQA",
    "google-research-datasets/nq_open": "NQOpen",
    "nq_open": "NQOpen",
    "squad_v2": "SQuADv2",
    "trivia_qa": "TriviaQA",
    "truthful_qa": "TruthfulQA",
    "gsm8k": "GSM8K",
}
