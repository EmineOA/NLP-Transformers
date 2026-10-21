"""Vérification légère de la présentation Gradio, sans chargement de Qwen."""
import unittest

from src.app.gradio_app import (
    build_app, format_verdict, ordered_concepts, render_course, render_progress, start_concept,
    submit_answer, toggle_model,
)
from src.app.gradio_theme import (
    CUSTOM_CSS, HEAD_HTML, HEADER_HTML, card_html, concept_tag, difficulty_label, make_theme,
    switch_html,
)


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

    def test_course_rendered_in_page_not_in_chat(self):
        state, chat = start_concept("monty-hall")
        course = render_course(state)
        self.assertTrue(course.startswith("## Introduction"))
        self.assertNotIn("Introduction", chat[0]["content"])
        self.assertEqual(render_course({"concept_id": None}), "")

    def test_new_sections_announced_in_chat(self):
        state, chat = start_concept("monty-hall")
        before = render_course(state)

        def perfect(*_args):
            return {"verdict": "PARFAIT", "feedback": "Oui.", "raw_response": "", "valid_format": True}

        state, chat = submit_answer("Je change.", state, chat, generate=perfect)
        self.assertGreater(len(render_course(state)), len(before))
        self.assertTrue(any("La suite du cours" in m["content"] for m in chat))

    def test_verdict_label_has_colour_class(self):
        self.assertIn('class="v PARTIEL"', format_verdict(
            {"verdict": "PARTIEL", "feedback": "x", "valid_format": True}))
        self.assertIn('class="v REFUS"', format_verdict(
            {"verdict": None, "feedback": "x", "valid_format": True}))

    def test_cards_follow_mockup_order(self):
        ids = [concept["id"] for concept in ordered_concepts()]
        self.assertEqual(ids[:2], ["monty-hall", "bateau-de-thesee"])
        self.assertEqual(len(ids), 6)
        self.assertIn("Commencer", card_html(ordered_concepts()[0]))

    def test_labels_and_switch(self):
        self.assertEqual([difficulty_label(n) for n in (25, 45, 75)], ["Facile", "Moyen", "Difficile"])
        self.assertEqual(concept_tag({"matiere": "philosophie", "difficulte": 35}), "Philo · Facile")
        self.assertEqual(toggle_model("baseline"), "lora")
        self.assertEqual(toggle_model("lora"), "baseline")
        self.assertIn('data-v="QLoRA"', switch_html("lora"))
        self.assertIn("@property --a", HEAD_HTML)

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
