"""Tests de l'export du CSV pédagogique vers la convention JSONL messages."""
import importlib
import json
import unittest
import csv
import tempfile
import subprocess
import sys
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

    def test_export_round_trip_and_reject_duplicate_ids(self):
        """Exporte les vrais cas du début du CSV et refuse les identifiants répétés."""
        export = importlib.import_module("data.dialogue.classification.export_jsonl")
        self.assertTrue(callable(getattr(export, "export_dataset", None)), "La fonction d'export manque.")
        with (ROOT / "data/dialogue/classification/dataset_source.csv").open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            fields, rows = reader.fieldnames, list(reader)[:4]
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "source.csv", Path(folder) / "dataset.jsonl"
            with source.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(export.export_dataset(source, output), len(rows))
            first = output.read_bytes()
            records = [json.loads(line) for line in first.decode("utf-8").splitlines()]
            self.assertEqual(len(records), len(rows))
            self.assertTrue(all(set(record) == {"messages"} for record in records))
            self.assertFalse(first.startswith(b"\xef\xbb\xbf"))
            export.export_dataset(source, output)
            self.assertEqual(first, output.read_bytes())
            with source.open("a", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=fields, delimiter=";").writerow(rows[0])
            with self.assertRaisesRegex(ValueError, "doubl"):
                export.export_dataset(source, output)
            self.assertEqual(first, output.read_bytes())

    def test_cli_exports_all_rows(self):
        """Exerce la commande publique sur le vrai CSV complet."""
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "complete.jsonl"
            result = subprocess.run([sys.executable, str(MODULE), "--output", str(output)],
                                    capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_file(), "La commande n'a produit aucun JSONL.")
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1003)
            self.assertIn("1003", result.stdout)

    def test_validator_reads_filled_csv_without_raw_json(self):
        """Le CSV fusionné reste vérifiable après retrait des sorties brutes."""
        import contextlib
        import io
        from unittest.mock import patch
        validation = importlib.import_module("data.dialogue.classification.valider_dataset")
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()) as output:
            with patch.object(validation, "GENERATION", folder), patch.object(sys, "argv", ["valider_dataset.py"]):
                validation.main()
            self.assertIn("1003 lignes contrôlées", output.getvalue())
            self.assertIn("'OK': 1003", output.getvalue())


if __name__ == "__main__":
    unittest.main()
