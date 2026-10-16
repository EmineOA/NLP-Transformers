"""Tests de l'export du CSV pédagogique vers la convention JSONL messages."""
import importlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "data/dialogue/classification/export_jsonl.py"


class TestDatasetExport(unittest.TestCase):
    def test_history_is_kept_without_inventing_turns(self):
        """Vérifie le contexte résumé, la cible séparée et l'absence de fuite."""
        self.assertTrue(MODULE.is_file(), "Le convertisseur export_jsonl.py manque.")
        export = importlib.import_module("data.dialogue.classification.export_jsonl")
        concept = json.loads((ROOT / "data/concepts/json/monty-hall.json").read_text(encoding="utf-8"))
        question = concept["questions"][1]
        row = {
            "id": "test_export_parfait", "concept_id": concept["id"],
            "question_id": question["id"], "historique": "[monty-hall_q1] Ancienne réponse test.",
            "essai_question": "2", "reponse_utilisateur": "Réponse courante test.",
            "label": "PARFAIT", "feedback": "Retour cible réservé au test.",
            "cas": "NE_PAS_INJECTER_CAS", "style": "NE_PAS_INJECTER_STYLE",
            "labels_historique": "NE_PAS_INJECTER_LABELS", "generateur": "NE_PAS_INJECTER_MODELE",
        }
        messages = export.build_messages(row, concept, question)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
        self.assertIn(row["historique"], messages[1]["content"])
        self.assertTrue(messages[1]["content"].endswith(row["reponse_utilisateur"]))
        self.assertIn(question["indices"][0], messages[1]["content"])
        self.assertNotIn(question["indices"][1], messages[1]["content"])
        self.assertEqual(messages[-1]["content"], "[VERDICT: PARFAIT]\n" + row["feedback"])
        inputs = messages[0]["content"] + messages[1]["content"]
        self.assertNotIn(row["feedback"], inputs)
        self.assertNotIn(row["id"], inputs)
        self.assertNotIn("NE_PAS_INJECTER", inputs)


if __name__ == "__main__":
    unittest.main()
