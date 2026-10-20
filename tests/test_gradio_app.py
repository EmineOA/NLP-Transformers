"""Tests de la logique du parcours Gradio (sans modèle ni gradio)."""
import unittest

from src.app.gradio_app import (
    MAX_ATTEMPTS,
    format_verdict,
    list_concepts,
    reset_session,
    start_concept,
    submit_answer,
)


def fake_generate(verdict):
    """Fabrique une fausse fonction d'inférence qui renvoie toujours ce verdict."""
    def generate(messages, concept, question, model_version="baseline"):
        raw = f"[VERDICT: {verdict}]\nRetour." if verdict else "texte sans tag"
        return {"verdict": verdict, "feedback": "Retour.", "raw_response": raw, "valid_format": bool(verdict)}
    return generate


class TestGradioApp(unittest.TestCase):
    def test_list_concepts(self):
        self.assertIn("monty-hall", [concept_id for _, concept_id in list_concepts()])

    def test_start_concept(self):
        state, chat = start_concept("monty-hall")
        self.assertEqual((state["question_index"], state["attempts"]), (0, 0))
        self.assertIn("Question 1/3", chat[-1]["content"])

    def test_parfait_advances(self):
        state, chat = start_concept("monty-hall")
        state, chat = submit_answer("ok", state, chat, generate=fake_generate("PARFAIT"))
        self.assertEqual(state["question_index"], 1)
        self.assertIn("✅ Parfait", format_verdict({"verdict": "PARFAIT", "feedback": "x", "valid_format": True}))

    def test_three_failures_reveal_key_points_then_advance(self):
        state, chat = start_concept("monty-hall")
        for _ in range(MAX_ATTEMPTS):
            state, chat = submit_answer("faux", state, chat, generate=fake_generate("INCORRECT"))
        self.assertEqual(state["question_index"], 1)
        self.assertTrue(any("ce qui était attendu" in m["content"] for m in chat))

    def test_invalid_output_does_not_advance(self):
        state, chat = start_concept("monty-hall")
        state, chat = submit_answer("?", state, chat, generate=fake_generate(None))
        self.assertEqual((state["question_index"], state["attempts"]), (0, 0))

    def test_end_of_concept_and_reset(self):
        state, chat = start_concept("monty-hall")
        for _ in range(3):
            state, chat = submit_answer("ok", state, chat, generate=fake_generate("PARFAIT"))
        self.assertTrue(state["finished"])
        self.assertIsNone(reset_session()[0]["concept_id"])


if __name__ == "__main__":
    unittest.main()
