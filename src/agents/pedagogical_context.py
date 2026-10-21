"""Format commun des messages de l'étudiant pour le dataset et l'application."""
from __future__ import annotations


def build_student_message(answer: str, history: str = "", hints_seen: list[str] | None = None) -> str:
    """Construit une entrée sans injecter le verdict ni le feedback attendus."""
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("La réponse doit être un texte non vide.")
    if not isinstance(history, str):
        raise ValueError("Le résumé de l'historique doit être du texte.")
    if hints_seen is not None and (not isinstance(hints_seen, list) or any(not isinstance(x, str) for x in hints_seen)):
        raise ValueError("Les indices doivent être une liste de textes.")
    parts = []
    if history:
        parts.append("HISTORIQUE RÉSUMÉ (contexte, pas la réponse à évaluer) :\n" + history)
    if hints_seen:
        parts.append("INDICES DÉJÀ FOURNIS (selon le numéro d'essai) :\n" + "\n".join(hints_seen))
    parts.append("RÉPONSE ACTUELLE À ÉVALUER :\n" + answer)
    return "\n\n".join(parts)


def summarize_history(completed_answers: list[dict], attempts: list[dict], current_question_id: str) -> str:
    """Conserve les réponses finales et les tentatives évaluées de la question."""
    lines = [f"[{item['question_id']}] {item['response']}" for item in completed_answers]
    for index, item in enumerate(attempts, 1):
        lines.append(
            f"[{current_question_id} – essai {index}] {item['response']} "
            f"→ [VERDICT: {item['verdict']}]"
        )
    return "\n".join(lines)
