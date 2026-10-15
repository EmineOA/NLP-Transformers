"""Bloc D / T6.1 : banc de tests reproductible sans dépendance ML.

Le calcul est séparé de la génération. Les verdicts attendus ne sont jamais
transmis au modèle. Les données de démonstration ne constituent pas des
mesures réelles de Brainmaxing.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

VERDICTS = {"PARFAIT", "PARTIEL", "INCORRECT"}
TAG = re.compile(r"\[VERDICT:\s*(PARFAIT|PARTIEL|INCORRECT)\]", re.I)


def parse_response(raw_response: str) -> dict:
    """Cas nominal : un UNIQUE tag autorisé en PREMIÈRE ligne.

    Une réponse hors périmètre sans tag ne peut pas être distinguée d'une
    génération mal formée à partir du texte seul. Elle doit être prise en charge
    par le contrat d'inférence commun, en amont du scoreur.
    """
    lines = raw_response.splitlines()
    first = lines[0].strip() if lines else ""
    match = TAG.fullmatch(first)
    all_tags = TAG.findall(raw_response)
    valid = bool(match) and len(all_tags) == 1
    return {
        "verdict": match.group(1).upper() if valid else None,
        "valid_format": valid,
        "feedback": "\n".join(lines[1:]).strip() if valid else raw_response,
        "raw_response": raw_response,
    }


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    data = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n} est un JSON invalide") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{n} doit contenir un objet")
            data.append(obj)
    return data


def planned_tests(cases: list[dict]) -> list[dict]:
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Identifiants de scénarios dupliqués")
    rows = []
    for case in cases:
        if case["type"] == "parser":
            rows.append({"key": (case["id"], 1), "type": "parser", "expected": case["expected"]})
        elif case["type"] == "conversation":
            for n, turn in enumerate(case["turns"], 1):
                rows.append({"key": (case["id"], n), "type": "conversation", "expected": turn["expected"]})
        else:
            raise ValueError(f"Type inconnu pour {case['id']}")
    return rows


def rate(n: int, total: int) -> float | None:
    return round(n / total, 4) if total else None


def score_suite(cases: list[dict], predictions: list[dict]) -> tuple[list[dict], dict]:
    """Une prédiction doit contenir id, tour, raw_response, verdict, valid_format.

    Pour un refus : verdict=null, valid_format=true (validé par l'orchestrateur),
    et refus_effectif=true/false seulement APRÈS relecture indépendante.
    Sans relecture, on ne prétend pas que le refus a été correct.
    """
    planned = planned_tests(cases)
    required = {r["key"] for r in planned}
    actual = {}
    for p in predictions:
        key = (p.get("id"), p.get("tour", 1))
        if key in actual:
            raise ValueError(f"Prédiction dupliquée : {key}")
        if not isinstance(p.get("valid_format"), bool):
            raise ValueError(f"valid_format booléen obligatoire pour {key}")
        if p.get("verdict") not in VERDICTS | {None}:
            raise ValueError(f"verdict non autorisé pour {key}")
        if not isinstance(p.get("raw_response"), str):
            raise ValueError(f"raw_response texte obligatoire pour {key}")
        if "refus_effectif" in p and not isinstance(p["refus_effectif"], bool):
            raise ValueError(f"refus_effectif doit être booléen pour {key}")
        actual[key] = p
    if actual.keys() != required:
        raise ValueError(f"Écart dans les IDs/tours : manquants={sorted(required-actual.keys())}, en trop={sorted(actual.keys()-required)}")

    details = []
    for case in planned:
        ident, tour = case["key"]
        p = actual[(ident, tour)]
        expected = case["expected"]
        observed_verdict = p["verdict"]
        observed_format = p["valid_format"]
        if case["type"] == "parser":
            parsed = parse_response(p["raw_response"])
            if observed_verdict != parsed["verdict"] or observed_format != parsed["valid_format"]:
                raise ValueError(f"Résultat du parseur incohérent pour {ident}")
        elif observed_verdict is not None:
            parsed = parse_response(p["raw_response"])
            if observed_verdict != parsed["verdict"] or observed_format != parsed["valid_format"]:
                raise ValueError(f"Tag/verdict/format incohérents pour {ident}, tour {tour}")
        elif observed_format and not expected.get("hors_perimetre", False):
            # Aucun verdict sur un cas nominal ne peut compter comme format valide.
            observed_format = False
        v_expected = expected.get("verdict")
        is_refusal = bool(expected.get("hors_perimetre", False))
        row = {
            "id": ident, "tour": tour, "type": case["type"],
            "attendu": v_expected, "observe": observed_verdict,
            "format_valide": observed_format,
            "verdict_correct": (observed_verdict == v_expected and observed_format) if v_expected in VERDICTS else None,
            "refus_formel": (observed_verdict is None and observed_format) if is_refusal else None,
            "refus_effectif_revu": p.get("refus_effectif") if is_refusal else None,
            "sortie_invalide_detectee": (not observed_format) if case["type"] == "parser" else None,
            "feedback": p.get("feedback", ""),
            "raw_response": p["raw_response"],
        }
        details.append(row)
    graded = [r for r in details if r["verdict_correct"] is not None]
    generated = [r for r in details if r["type"] == "conversation"]
    refusal = [r for r in details if r["refus_formel"] is not None]
    reviewed = [r for r in refusal if r["refus_effectif_revu"] is not None]
    parser = [r for r in details if r["type"] == "parser"]
    metrics = {
        "nombre_scenarios": len(cases),
        "nombre_tours_evalues": len(details),
        "accuracy_verdict": rate(sum(r["verdict_correct"] for r in graded), len(graded)),
        "n_verdicts": len(graded),
        "taux_format_conforme": rate(sum(r["format_valide"] for r in generated), len(generated)),
        "n_requetes": len(generated),
        "taux_refus_formels": rate(sum(r["refus_formel"] for r in refusal), len(refusal)),
        "n_demandes_refus": len(refusal),
        "taux_refus_effectifs_revus": rate(sum(r["refus_effectif_revu"] for r in reviewed), len(reviewed)),
        "n_refus_relus": len(reviewed),
        "taux_detection_format_invalide": rate(sum(r["sortie_invalide_detectee"] for r in parser), len(parser)),
        "n_tests_parseur": len(parser),
    }
    return details, metrics


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--cases", default="evaluation/cases_v0.1.jsonl")
    cli.add_argument("--predictions", required=True)
    cli.add_argument("--output", default="evaluation/eval_v0.1_baseline.csv")
    cli.add_argument("--versions", default="evaluation/versions_v0.1.json")
    args = cli.parse_args()
    cases = load_jsonl(args.cases)
    predictions = load_jsonl(args.predictions)
    details, metrics = score_suite(cases, predictions)
    if not Path(args.versions).exists():
        raise FileNotFoundError(f"Renseigner les versions utilisées : {args.versions}")
    versions = json.loads(Path(args.versions).read_text(encoding="utf-8"))
    if versions.get("origine_predictions") == "simulation":
        print("ATTENTION : données FICTIVES de démonstration ; scores non attribuables à Qwen.")
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=details[0].keys())
        writer.writeheader()
        writer.writerows(details)
    print(json.dumps({"versions": versions, "metriques": metrics}, ensure_ascii=False, indent=2))
    print("CSV :", target)


if __name__ == "__main__":
    main()
