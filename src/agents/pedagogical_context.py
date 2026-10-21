"""Convention commune de messages pour le dataset et l'interface pédagogique.

Le texte attendu du modèle (verdict ou feedback cible) ne doit jamais figurer
dans le message d'entrée utilisé pour l'inférence.
"""
from __future__ import annotations


def build_student_message(answer: str, history: str = "", hints_seen: list[str] | None = None) -> str:
    """Assemble le message utilisateur selon la convention du dataset QLoRA.

    Paramètres :
        answer : réponse de l'étudiant à la question courante.
        history : résumé des réponses antérieures du même concept.
        hints_seen : indices affichés après les essais précédents.

    Retour :
        entrée structurée destinée au modèle, sans réponse attendue.
    """
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("La réponse courante doit être un texte non vide.")
    chunks = []
    if history:
        chunks.append("HISTORIQUE RÉSUMÉ (contexte, pas la réponse à évaluer) :\n" + history)
    if hints_seen:
        chunks.append("INDICES DÉJÀ FOURNIS (selon le numéro d'essai) :\n" + "\n".join(hints_seen))
    chunks.append("RÉPONSE ACTUELLE À ÉVALUER :\n" + answer)
    return "\n\n".join(chunks)


def summarize_history(
    completed_answers: list[dict],
    attempts: list[dict],
    current_question_id: str,
) -> str:
    """Conserve les réponses finales des questions et les essais de la question courante.

    Paramètres :
        completed_answers : dictionnaires avec question_id et response.
        attempts : dictionnaires avec response et verdict.
        current_question_id : identifiant de la question en cours.

    Retour :
        résumé stable, indépendant de la fenêtre des derniers tours du LLM.
    """
    lines = [
        f"[{item['question_id']}] {item['response']}"
        for item in completed_answers
    ]
    for number, item in enumerate(attempts, start=1):
        lines.append(
            f"[{current_question_id} – essai {number}] {item['response']} "
            f"→ [VERDICT: {item['verdict']}]"
        )
    return "\n".join(lines)
