"""Tests automatiques de T6.1 : pas d'appel modèle, pas de GPU."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score import load_jsonl, parse_response, planned_tests, score_suite

PATH = Path(__file__).parent


def fake_predictions(cases):
    """Prédictions volontairement FABRIQUÉES pour tester la logique du scoreur."""
    results = []
    for case in cases:
        if case["type"] == "parser":
            raw = case["raw_response"]
            p = parse_response(raw)
            results.append({"id": case["id"], "tour": 1, **p})
            continue
        for n, t in enumerate(case["turns"], 1):
            verdict = t["expected"]["verdict"]
            if verdict is None:
                results.append({"id": case["id"], "tour": n, "verdict": None,
                                "valid_format": True, "feedback": "Je ne peux pas répondre à cette demande.",
                                "raw_response": "Je ne peux pas répondre à cette demande.",
                                "refus_effectif": True})
            else:
                raw = f"[VERDICT: {verdict}]\nRetour pédagogique factice."
                results.append({"id": case["id"], "tour": n, **parse_response(raw)})
    return results


class TestT61(unittest.TestCase):
    def setUp(self):
        self.cases = load_jsonl(PATH / "cases_v0.1.jsonl")

    def test_nombre_scenarios_et_tours(self):
        self.assertEqual(len(self.cases), 8)
        self.assertEqual(len(planned_tests(self.cases)), 9)
        self.assertEqual([c["id"] for c in self.cases], ["N1", "N2", "N3", "N4", "N5", "X1", "X2", "X3"])
        self.assertEqual(len(self.cases[3]["turns"]), 2)

    def test_tags_en_premiere_ligne(self):
        p = parse_response("[VERDICT: partiel]\nIndice utile")
        self.assertTrue(p["valid_format"])
        self.assertEqual(p["verdict"], "PARTIEL")
        self.assertFalse(parse_response("Indice utile\n[VERDICT: PARTIEL]")["valid_format"])
        self.assertFalse(parse_response("[VERDICT: PARFAIT]\n[VERDICT: PARTIEL]")["valid_format"])
        self.assertFalse(parse_response("Bravo, sans tag")["valid_format"])

    def test_fiches_et_questions_existantes(self):
        import json
        root = PATH.parent / "data" / "concepts" / "json"
        if not root.exists():
            self.skipTest("Copier le dossier evaluation à la racine du dépôt avant validation des fiches")
        for case in self.cases:
            if case["type"] == "parser" or case["id"] == "X2":
                continue
            fiche = json.loads((root / f"{case['concept_id']}.json").read_text(encoding="utf-8"))
            self.assertIn(case["question_id"], {q["id"] for q in fiche["questions"]})
        self.assertFalse((root / "paradoxe-de-russell.json").exists())

    def test_scores_fictifs(self):
        preds = fake_predictions(self.cases)
        _, scores = score_suite(self.cases, preds)
        self.assertEqual(scores["n_verdicts"], 5)
        self.assertEqual(scores["n_demandes_refus"], 3)
        self.assertEqual(scores["accuracy_verdict"], 1.0)
        self.assertEqual(scores["taux_refus_formels"], 1.0)
        self.assertEqual(scores["taux_detection_format_invalide"], 1.0)

    def test_une_erreur_de_grade_reduit_accuracy(self):
        preds = fake_predictions(self.cases)
        preds[0] = {"id": "N1", "tour": 1, **parse_response("[VERDICT: PARTIEL]\nTest factice")}
        _, scores = score_suite(self.cases, preds)
        self.assertEqual(scores["accuracy_verdict"], 0.8)

    def test_refus_non_relu_pas_note_comme_correct(self):
        preds = fake_predictions(self.cases)
        for p in preds:
            p.pop("refus_effectif", None)
        _, scores = score_suite(self.cases, preds)
        self.assertIsNone(scores["taux_refus_effectifs_revus"])
        self.assertEqual(scores["n_refus_relus"], 0)

    def test_sortie_validee_incoherente_interdite(self):
        preds = fake_predictions(self.cases)
        preds[0]["verdict"] = "PARTIEL"
        with self.assertRaises(ValueError):
            score_suite(self.cases, preds)

    def test_ids_manquants_interdits(self):
        preds = fake_predictions(self.cases)
        with self.assertRaises(ValueError):
            score_suite(self.cases, preds[:-1])


if __name__ == "__main__":
    unittest.main()
