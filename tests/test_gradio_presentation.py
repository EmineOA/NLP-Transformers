"""Vérification légère de la présentation Gradio, sans chargement de Qwen."""
import unittest

from src.app.gradio_app import build_app, render_progress, start_concept, submit_answer
from src.app.gradio_theme import CUSTOM_CSS, HEADER_HTML, make_theme


class TestGradioPresentation(unittest.TestCase):
    def test_progress_initial(self):
        output = render_progress({"concept_id": None})
        self.assertIn("Prêt à commencer", output)
        self.assertIn('aria-valuenow="0"', output)

    def test_progress_after_selection(self):
        state, _ = start_concept("monty-hall")
        output = render_progress(state)
        self.assertIn("Question 1 / 3", output)
        self.assertIn("Tentative 1 / 3", output)

    def test_progress_updates_after_question(self):
        state, chat = start_concept("monty-hall")

        def perfect(*_args):
            return {
                "verdict": "PARFAIT",
                "feedback": "Raisonnement valide.",
                "raw_response": "[VERDICT: PARFAIT]\nRaisonnement valide.",
                "valid_format": True,
            }

        state, _ = submit_answer("Je justifie.", state, chat, generate=perfect)
        output = render_progress(state)
        self.assertIn("Question 2 / 3", output)
        self.assertIn('aria-valuenow="33"', output)

    def test_accessible_visuals(self):
        self.assertIn("Brain", HEADER_HTML)
        self.assertIn("prefers-reduced-motion", CUSTOM_CSS)
        self.assertIn("@media (max-width: 900px)", CUSTOM_CSS)
        self.assertIn("bm-progress-card", CUSTOM_CSS)

    def test_builds_without_loading_model(self):
        try:
            import gradio as gr
        except ImportError:
            self.skipTest("Gradio est facultatif pour les tests métier.")
        app = build_app()
        self.assertIsNotNone(app)
        self.assertIsNotNone(make_theme(gr))


if __name__ == "__main__":
    unittest.main()
