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

    def test_dark_palette_covers_both_gradio_modes(self):
        try:
            import gradio as gr
        except ImportError:
            self.skipTest("Gradio facultatif pour les tests métier.")
        theme = make_theme(gr)
        for variable in (
            "body_background_fill",
            "block_background_fill",
            "block_label_background_fill",
            "block_label_text_color",
            "block_title_text_color",
            "input_background_fill",
            "button_secondary_background_fill",
            "button_secondary_text_color",
        ):
            self.assertEqual(
                getattr(theme, variable),
                getattr(theme, variable + "_dark"),
                variable,
            )

    def test_dark_theme_contrast(self):
        try:
            import gradio as gr
        except ImportError:
            self.skipTest("Gradio facultatif pour les tests métier.")
        theme = make_theme(gr)

        def luminance(hex_color):
            rgb = [int(hex_color[i:i+2], 16) / 255 for i in (1, 3, 5)]
            linear = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in rgb]
            return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

        for foreground, background in (
            ("block_label_text_color", "block_label_background_fill"),
            ("body_text_color", "body_background_fill"),
            ("button_secondary_text_color", "button_secondary_background_fill"),
        ):
            light, dark = sorted((luminance(getattr(theme, foreground)),
                                  luminance(getattr(theme, background))))
            contrast = (dark + 0.05) / (light + 0.05)
            self.assertGreaterEqual(contrast, 4.5, foreground)

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
