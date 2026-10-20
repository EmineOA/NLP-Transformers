"""Point d'entrée commun A/B vers les blocs D (évaluation), E (Gradio), C (LoRA).

Contrat de backlog.md : generate_response(messages, concept, question, model_version)
-> verdict / feedback / raw_response / valid_format.

Ce module ne tient AUCUN historique caché : l'appelant transmet les échanges.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from src.agents.baseline import parse_pedagogical_response
from src.agents.concepts import load_concept
from src.llm.generation import generate_chat
from src.llm.prompts import build_pedagogical_prompt


@lru_cache(maxsize=1)
def _default_components():
    """Charge une seule fois le modèle pour les appels successifs (hors tests)."""
    from src.llm.model import load_model
    return load_model()


# Meilleur adaptateur du run final (v1.1, nouveau split train/val/test). Le dossier adapters/ est ignoré
# par git : on le télécharge depuis Hugging Face (MonsieurNikko/brainmaxing-qlora-v1.1).
LORA_ADAPTER_DIR = Path(__file__).resolve().parents[2] / "adapters" / "v1.1" / "best"


@lru_cache(maxsize=1)
def _lora_components():
    """Charge une seule fois Qwen 4 bits avec l'adaptateur LoRA du bloc C.

    Retour :
        LLMComponents (tokenizer, model, device), comme _default_components.

    Exemple :
        _lora_components().device  # device(type='cuda')
    """
    # Vérifié avant tout chargement pour échouer vite avec un message clair.
    if not LORA_ADAPTER_DIR.is_dir():
        raise FileNotFoundError(
            f"Adaptateur LoRA introuvable : copier le dossier best/ de Drive dans {LORA_ADAPTER_DIR}"
        )

    # Imports locaux : peft et bitsandbytes ne sont pas nécessaires pour la baseline.
    from src.llm.model import LLMComponents, get_device
    from src.training.model import load_model_with_adapter

    model, tokenizer = load_model_with_adapter(str(LORA_ADAPTER_DIR))
    # load_model_4bit exige CUDA et place le modèle sur le GPU 0 (device_map={"": 0}) ;
    # get_device() renvoie alors "cuda", c'est-à-dire le GPU courant, le 0 par défaut.
    return LLMComponents(tokenizer=tokenizer, model=model, device=get_device())


def _controlled_refusal(feedback: str) -> dict:
    """Produit un refus explicite hors génération, dans le contrat commun."""
    raw = f"[HORS_PERIMETRE]\n{feedback}"
    return parse_pedagogical_response(raw)


def _resolve_question(concept: dict, question: dict | str | None) -> dict | None:
    """Accepte uniquement une question appartenant au concept sélectionné."""
    identifier = question.get("id") if isinstance(question, dict) else question
    for candidate in concept["questions"]:
        if candidate["id"] == identifier:
            return candidate
    return None


def _validate_messages(messages: list[dict]) -> list[dict]:
    """Vérifie le format conversationnel sans accepter de 'system' fourni par l'UI."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages doit contenir au moins un message utilisateur.")
    checked = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            raise ValueError("Seuls les messages 'user' et 'assistant' sont acceptés.")
        if not isinstance(message.get("content"), str):
            raise ValueError("Le contenu d'un message doit être du texte.")
        checked.append({"role": message["role"], "content": message["content"]})
    if checked[-1]["role"] != "user":
        raise ValueError("Le dernier message doit provenir de l'étudiant.")
    return checked


def _bounded_history(messages: list[dict], max_history_turns: int) -> list[dict]:
    """Garde au plus N échanges complets et la question utilisateur courante."""
    if max_history_turns < 0:
        raise ValueError("max_history_turns doit être positif ou nul.")
    older = messages[:-1]
    # Prendre les derniers N couples user/assistant complets seulement.
    turns = []
    index = len(older) - 1
    while index >= 1 and len(turns) < max_history_turns:
        if older[index]["role"] == "assistant" and older[index - 1]["role"] == "user":
            turns.append(older[index - 1:index + 1])
            index -= 2
        else:
            index -= 1
    return [message for pair in reversed(turns) for message in pair] + [messages[-1]]


def generate_response(
    messages: list[dict],
    concept: dict | str,
    question: dict | str | None,
    model_version: str = "baseline",
    *,
    prompt_version: str = "fewshot",
    max_history_turns: int = 3,
    components: Any = None,
    max_new_tokens: int = 160,
) -> dict:
    """Évalue une réponse de l'étudiant (baseline ou LoRA), selon la grille du bloc A.

    Paramètres :
        messages : échanges au format role/content, terminés par un message user.
        concept : fiche JSON ou ID de fiche à charger.
        question : objet question ou ID de question de la fiche.
        model_version : 'baseline' (Qwen en 16 bits) ou 'lora' (Qwen en 4 bits + adaptateur du bloc C).
            Appeler les deux dans le même processus garde les deux modèles en mémoire GPU.
        prompt_version : 'criteria' ou 'fewshot' (comparaison de prompts).
        max_history_turns : nombre maximal de tours complets précédents.
        components : objet model/tokenizer/device facultatif, pour réutilisation et tests.
            S'il est fourni, il est utilisé tel quel, quel que soit model_version.

    Retour : dictionnaire du contrat 2. Une sortie de Qwen mal formatée reste
    invalide : aucun verdict n'est inventé ou corrigé artificiellement.
    """
    cleaned = _validate_messages(messages)
    if model_version not in {"baseline", "lora"}:
        raise ValueError("model_version doit valoir 'baseline' ou 'lora'.")
    if isinstance(concept, str):
        try:
            concept = load_concept(concept)
        except (FileNotFoundError, ValueError):
            return _controlled_refusal("Ce concept ne figure pas dans le corpus pédagogique disponible.")
    if not isinstance(concept, dict) or not isinstance(concept.get("questions"), list):
        raise ValueError("Fiche concept invalide.")
    selected = _resolve_question(concept, question)
    if selected is None:
        return _controlled_refusal("Cette question n'appartient pas au concept sélectionné.")

    bounded = _bounded_history(cleaned, max_history_turns)
    prompt = build_pedagogical_prompt(concept, selected, version=prompt_version)
    if components is not None:
        llm = components
    elif model_version == "lora":
        llm = _lora_components()
    else:
        llm = _default_components()
    raw = generate_chat(
        model=llm.model,
        tokenizer=llm.tokenizer,
        messages=[{"role": "system", "content": prompt}] + bounded,
        device=llm.device,
        max_new_tokens=max_new_tokens,
    )
    return parse_pedagogical_response(raw)
