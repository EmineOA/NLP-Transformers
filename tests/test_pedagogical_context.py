"""Tests sans GPU du contexte partagé entre l'export QLoRA et Gradio."""
import unittest

from src.agents.pedagogical_context import build_student_message, summarize_history


class TestPedagogicalContext(unittest.TestCase):
    def test_minimal_message(self):
        self.assertEqual(
            build_student_message("Une réponse."),
            "RÉPONSE ACTUELLE À ÉVALUER :\nUne réponse.",
        )

    def test_message_with_history_and_hint(self):
        self.assertEqual(
            build_student_message("Une réponse.", "[q1] Première réponse", ["Indice A"]),
            "HISTORIQUE RÉSUMÉ (contexte, pas la réponse à évaluer) :\n"
            "[q1] Première réponse\n\n"
            "INDICES DÉJÀ FOURNIS (selon le numéro d'essai) :\n"
            "Indice A\n\n"
            "RÉPONSE ACTUELLE À ÉVALUER :\nUne réponse.",
        )

    def test_summary_preserves_previous_questions_and_current_attempts(self):
        self.assertEqual(
            summarize_history(
                [{"question_id": "q1", "response": "Réponse Q1"}],
                [{"response": "Premier essai Q2", "verdict": "PARTIEL"}],
                "q2",
            ),
            "[q1] Réponse Q1\n[q2 – essai 1] Premier essai Q2 → [VERDICT: PARTIEL]",
        )

    def test_reject_invalid_input(self):
        with self.assertRaises(ValueError):
            build_student_message(" ")
        with self.assertRaises(ValueError):
            build_student_message("réponse", history=None)
        with self.assertRaises(ValueError):
            build_student_message("réponse", hints_seen=["valide", None])


if __name__ == "__main__":
    unittest.main()
