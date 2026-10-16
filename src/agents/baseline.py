"""Compatibilité expérimentale et agent pédagogique avec historique explicite."""
import re

from src.agents.history import ConversationHistory
from src.llm.generation import generate_chat
from src.llm.prompts import build_system_prompt

# Mode historique des premiers notebooks (à ne pas mélanger au domaine pédagogique).
VERDICT_PATTERN = re.compile(r"\[VERDICT:\s*([^\]]+)\]", re.IGNORECASE)
OLD_VERDICTS = {"ADMISSIBLE", "NON_ADMISSIBLE", "INDETERMINE"}

# Contrat A/B : à faire valider par l'équipe pour le refus brut.
VERDICT_TAG = re.compile(r"\[VERDICT:\s*(PARFAIT|PARTIEL|INCORRECT)\]", re.I)
REFUSAL_TAG = "[HORS_PERIMETRE]"
ANY_TAG = re.compile(r"\[VERDICT:[^\]]*\]|\[HORS_PERIMETRE\]", re.I)


def extract_verdict(response: str) -> str | None:
    """Préserve les anciens essais du notebook sur l'admissibilité."""
    match = VERDICT_PATTERN.search(response)
    result = match.group(1).strip().upper() if match else None
    return result if result in OLD_VERDICTS else None


def parse_pedagogical_response(raw_response: str) -> dict:
    """Parse strictement la première ligne et le feedback de la réponse.

    Un refus valide : verdict=None, valid_format=True, feedback non vide.
    Une sortie incorrecte : verdict=None, valid_format=False. Aucun rattrapage
    automatique du modèle n'est fait ici, afin de mesurer les vraies erreurs.
    """
    if not isinstance(raw_response, str):
        raise TypeError("raw_response doit être une chaîne de caractères.")
    lines = raw_response.strip().splitlines()
    first = lines[0].strip() if lines else ""
    feedback = "\n".join(lines[1:]).strip()
    match = VERDICT_TAG.fullmatch(first)
    is_refusal = first == REFUSAL_TAG
    # Rejeter les tags dupliqués, même si la première ligne était correcte.
    valid = bool(feedback) and (bool(match) or is_refusal) and len(ANY_TAG.findall(raw_response)) == 1
    return {
        "verdict": match.group(1).upper() if match and valid else None,
        "feedback": feedback if valid else "",
        "raw_response": raw_response,
        "valid_format": bool(valid),
    }


class BaselineAgent:
    """Agent pratique pour les notebooks ; délègue l'inférence pédagogique à B."""

    def __init__(
        self, model, tokenizer, device,
        rules: str | None = None,
        fiche: str | dict | None = None,
        max_history_turns: int = 3,
        system_prompt: str | None = None,
        question: dict | None = None,
        prompt_version: str = "fewshot",
    ):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.pedagogical = question is not None
        self.question, self.fiche = question, fiche
        self.max_history_turns = max_history_turns
        self.prompt_version = prompt_version
        self.history = ConversationHistory(max_turns=max_history_turns)
        if self.pedagogical:
            if not isinstance(fiche, dict) or not isinstance(question, dict) or not any(
                q["id"] == question.get("id") for q in fiche.get("questions", [])
            ):
                raise ValueError("La question doit appartenir à la fiche JSON.")
            if system_prompt is not None:
                raise ValueError("Utiliser prompt_version pour comparer les prompts pédagogiques.")
            self.system_prompt = None  # Recalculé pour chaque question par inference.py.
        else:
            if not isinstance(rules, str) or not isinstance(fiche, str):
                raise ValueError("Le mode historique attend rules et fiche sous forme de texte.")
            self.system_prompt = system_prompt or build_system_prompt(rules, fiche)

    def ask(self, user_message: str) -> dict:
        """Génère la réponse et ajoute ce tour à l'historique borné."""
        history = self.history.get_messages()
        user = {"role": "user", "content": user_message}
        if self.pedagogical:
            # Import différé : évite une dépendance circulaire au parseur.
            from types import SimpleNamespace
            from src.agents.inference import generate_response
            result = generate_response(
                history + [user], self.fiche, self.question,
                components=SimpleNamespace(model=self.model, tokenizer=self.tokenizer, device=self.device),
                max_history_turns=self.max_history_turns,
                prompt_version=self.prompt_version,
            )
            raw = result["raw_response"]
        else:
            raw = generate_chat(
                model=self.model, tokenizer=self.tokenizer,
                device=self.device,
                messages=[{"role": "system", "content": self.system_prompt}] + history + [user],
            )
            result = {"response": raw, "verdict": extract_verdict(raw)}
        self.history.add_user(user_message)
        self.history.add_assistant(raw)
        return result

    def set_question(self, question: dict) -> None:
        """Change de question sans effacer l'historique du concept courant."""
        if not self.pedagogical or not isinstance(question, dict) or not any(
            q["id"] == question.get("id") for q in self.fiche["questions"]
        ):
            raise ValueError("Question absente de la fiche pédagogique active.")
        self.question = question

    def reset(self) -> None:
        """Efface les tours en mémoire (nouveau concept ou nouvelle session)."""
        self.history.clear()
