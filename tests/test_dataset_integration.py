"""Tests techniques d'intégration ; backend simulé, aucun score de Qwen."""
import csv
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.agents.baseline import parse_pedagogical_response
from src.agents.concepts import load_concept, load_question
from src.agents.inference import generate_response

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data/dialogue/classification"


class TestDatasetIntegration(unittest.TestCase):
    def test_all_exported_inputs_reach_main_engine_unchanged(self):
        """Vérifie les 1 003 cibles et l'acheminement des entrées sans lancer Qwen."""
        with (FOLDER / "dataset_source.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter=";"))
        records = [json.loads(line) for line in (FOLDER / "dataset_messages.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1003)
        self.assertEqual(len(records), len(rows))
        concepts = {row["concept_id"]: load_concept(row["concept_id"]) for row in rows}
        components = SimpleNamespace(model=object(), tokenizer=object(), device="cpu")
        with patch("src.agents.inference.generate_chat", return_value="SORTIE_SIMULEE_INVALIDE") as backend:
            for row, record in zip(rows, records):
                messages = record["messages"]
                parsed = parse_pedagogical_response(messages[-1]["content"])
                self.assertTrue(parsed["valid_format"], row["id"])
                self.assertEqual(parsed["verdict"], None if row["label"] == "HORS_PERIMETRE" else row["label"])
                self.assertEqual(parsed["feedback"], row["feedback"].strip())
                concept = concepts[row["concept_id"]]
                question = load_question(concept, row["question_id"])
                result = generate_response(messages[1:-1], concept, question, components=components)
                self.assertEqual(backend.call_args.kwargs["messages"], messages[:-1], row["id"])
                self.assertFalse(result["valid_format"])
            self.assertEqual(backend.call_count, len(rows))


if __name__ == "__main__":
    unittest.main()
