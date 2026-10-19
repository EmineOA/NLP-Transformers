"""Tests de la suite T6.2 gelée (ne téléchargent pas Qwen)."""
import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.agents.final_evaluation import (
    SELECTION, SOURCE, git_blob_sha,
    build_scenarios, create_frozen_cases, load_frozen_cases, load_selection,
    read_csv, validate_suite, HOLDOUT_QUESTIONS,
)
from evaluation.score import planned_tests, score_suite
from evaluation.run_final import evaluate


class TestFinalEvaluation(unittest.TestCase):
    def test_selection_and_holdout(self):
        selection = load_selection()
        self.assertEqual(len(selection["tests"]), 20)
        self.assertEqual(len(selection["dialogues"]), 3)
        self.assertEqual(git_blob_sha(SOURCE), selection["source_blob_sha"])
        _, rows = read_csv(SOURCE)
        by_id = {r["id"]: r for r in rows}
        self.assertEqual(len(rows), 1003)
        self.assertEqual({r["question_id"] for r in rows if r["split"] == "test"}, HOLDOUT_QUESTIONS)
        for test in selection["tests"]:
            self.assertEqual(by_id[test["source_id"]]["split"], "test", test["id"])

    def test_generate_20_cases_and_3_dialogues(self):
        selection = load_selection()
        with tempfile.TemporaryDirectory() as temp:
            csv_path = Path(temp) / "cases.csv"
            lock_path = Path(temp) / "lock.json"
            create_frozen_cases(output=csv_path, lock_path=lock_path)
            rows, selection = load_frozen_cases(csv_path, SELECTION, lock_path)
            self.assertEqual(len(rows), 20)
            self.assertEqual(Counter(r["label"] for r in rows), {
                "PARFAIT": 5, "PARTIEL": 5, "INCORRECT": 5, "HORS_PERIMETRE": 5,
            })
            scenarios = build_scenarios(rows, selection)
            self.assertEqual(len(scenarios), 17)
            self.assertEqual(len(planned_tests(scenarios)), 20)
            self.assertEqual(sum(len(s["turns"]) == 2 for s in scenarios), 3)
            # La question précédente est préservée, l'essai 1 n'est pas dupliqué
            # dans le message user du tour 2 (il est dans les rôles chat).
            c01 = next(s for s in scenarios if s["id"] == "C01")
            self.assertIn("bateau-de-thesee_q1", c01["turns"][1]["user_input"])
            self.assertNotIn("[bateau-de-thesee_q3 – essai 1]", c01["turns"][1]["user_input"])
            self.assertEqual(c01["turns"][0]["expected"]["verdict"], "INCORRECT")
            for row in rows:
                # Les targets ne sont pas injectées dans les messages user.
                turn = next(t for sc in scenarios for t in sc["turns"] if t["test_id"] == row["id"])
                self.assertNotIn(row["feedback"], turn["user_input"])

    def test_frozen_files_are_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            csv_path, lock_path = Path(temp) / "cases.csv", Path(temp) / "lock.json"
            create_frozen_cases(output=csv_path, lock_path=lock_path)
            with self.assertRaises(FileExistsError):
                create_frozen_cases(output=csv_path, lock_path=lock_path)
            with csv_path.open("a", encoding="utf-8") as stream:
                stream.write("modification_non_autorisee\n")
            with self.assertRaisesRegex(ValueError, "modifiée après gel"):
                load_frozen_cases(csv_path, SELECTION, lock_path)

    def test_mocked_inference_all_20_without_leaking_expected(self):
        with tempfile.TemporaryDirectory() as temp:
            csv_path, lock_path = Path(temp) / "cases.csv", Path(temp) / "lock.json"
            create_frozen_cases(output=csv_path, lock_path=lock_path)
            rows, selection = load_frozen_cases(csv_path, SELECTION, lock_path)
            scenarios = build_scenarios(rows, selection)
            dummy = SimpleNamespace(model=object(), tokenizer=object(), device="cpu")
            fake_result = "[VERDICT: PARTIEL]\nIl manque une justification. Peux-tu développer ?"
            with patch("src.agents.inference.generate_chat", return_value=fake_result) as backend:
                predictions, partial = evaluate(scenarios, components=dummy)
            self.assertFalse(partial)
            self.assertEqual(len(predictions), 20)
            self.assertEqual(backend.call_count, 20)
            details, metrics = score_suite(scenarios, predictions)
            self.assertEqual(len(details), 20)
            self.assertEqual(metrics["n_requetes"], 20)
            self.assertEqual(metrics["n_sorties_llm"], 20)
            # Une paire doit vraiment passer l'échange précédent au moteur.
            with patch("src.agents.inference.generate_chat", return_value=fake_result) as backend:
                one_pair, partial = evaluate(scenarios, only="T02", components=dummy)
                messages = backend.call_args.kwargs["messages"]
            self.assertTrue(partial)
            self.assertEqual(len(one_pair), 2)
            self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
            self.assertEqual(messages[2]["content"], fake_result)


if __name__ == "__main__":
    unittest.main()
