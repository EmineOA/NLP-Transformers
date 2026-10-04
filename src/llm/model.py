from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class LLMComponents:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: torch.device


def get_device() -> torch.device:
    """
    Sélectionne automatiquement le meilleur device disponible.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
) -> LLMComponents:
    """
    Charge le tokenizer et le modèle causal.

    Returns
    -------
    LLMComponents
        Contient le tokenizer, le modèle et le device utilisé.
    """

    device = get_device()

    print(f"Chargement du modèle : {model_name}")
    print(f"Device utilisé : {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    )

    model.to(device)
    model.eval()

    return LLMComponents(
        tokenizer=tokenizer,
        model=model,
        device=device,
    )