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

    def test_course_is_progressive(self):
        from src.agents.concepts import load_concept
        concept = load_concept("monty-hall")
        state, chat = start_concept("monty-hall")
        first = concept["questions"][0]["apres_section"]
        self.assertEqual(state["shown_sections"], first)
        for section in concept["explication"][first:]:
            self.assertNotIn(section["contenu"], chat[0]["content"])

    def test_hint_after_each_failure_and_context(self):
        from src.agents.concepts import load_concept
        concept = load_concept("monty-hall")
        question = concept["questions"][0]
        state, chat = start_concept("monty-hall")
        captured = []

        def generate(messages, concept, question, model_version="baseline"):
            captured.append(messages[-1]["content"])
            return {"verdict": "INCORRECT", "feedback": "Essaie autrement.",
                    "raw_response": "[VERDICT: INCORRECT]\\nEssaie autrement.",
                    "valid_format": True}

        state, chat = submit_answer("Premier essai", state, chat, generate=generate)
        self.assertIn(question["indices"][0], chat[-1]["content"])
        state, chat = submit_answer("Deuxième essai", state, chat, generate=generate)
        self.assertIn(question["indices"][1], chat[-1]["content"])
        self.assertIn(question["indices"][0], captured[1])
        self.assertIn("Premier essai", captured[1])
        self.assertNotIn(question["indices"][1], captured[1])
        state, chat = submit_answer("Troisième essai", state, chat, generate=generate)
        self.assertIn("ce qui était attendu", chat[-1]["content"])
        self.assertEqual(state["question_index"], 1)

    def test_previous_question_answer_survives_progression(self):
        state, chat = start_concept("bateau-de-thesee")
        inputs = []

        def generate(messages, concept, question, model_version="baseline"):
            inputs.append(messages[-1]["content"])
            return {"verdict": "PARFAIT", "feedback": "Argument cohérent.",
                    "raw_response": "[VERDICT: PARFAIT]\\nArgument cohérent.",
                    "valid_format": True}

        for text in ("Je défends la continuité.", "Le réparé reste le même.",
                     "Je choisis selon ma règle."):
            state, chat = submit_answer(text, state, chat, generate=generate)
        self.assertIn("[bateau-de-thesee_q1] Je défends la continuité.", inputs[2])
        self.assertIn("[bateau-de-thesee_q2] Le réparé reste le même.", inputs[2])

    def test_two_question_concept_and_reset(self):
        state, chat = start_concept("plier-une-feuille-42-fois")
        self.assertIn("Question 1/2", chat[-1]["content"])
        for _ in range(2):
            state, chat = submit_answer("Une réponse.", state, chat,
                                        generate=fake_generate("PARFAIT"))
        self.assertTrue(state["finished"])
        self.assertEqual(reset_session()[0]["completed_answers"], [])


if __name__ == "__main__":
    unittest.main()
