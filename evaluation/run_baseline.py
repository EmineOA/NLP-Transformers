"""Bloc D / T6.1 : exécute les scénarios via l'API commune du bloc B.

Les références « expected » restent dans le jeu de test pour le scoring ;
elles ne sont jamais transmises au modèle.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.score import load_jsonl, planned_tests, score_suite


def run_cases(cases: list[dict], *, prompt_version: str = "fewshot",
              max_history_turns: int = 3, max_new_tokens: int = 160) -> list[dict]:
    """Exécute les cas indépendamment via generate_response().

    Les tours d'un même scénario partagent leur historique ; deux scénarios
    distincts n'en partagent jamais. Les verdicts attendus sont exclus des appels.
    """
    from src.agents.baseline import parse_pedagogical_response
    from src.agents.concepts import load_concept
    from src.agents.inference import generate_response

    predictions = []
    for case in cases:
        if case["type"] == "parser":
            result = parse_pedagogical_response(case["raw_response"])
            predictions.append({"id": case["id"], "tour": 1,
                                "source": "injection_parseur", **result})
            print(f"{case['id']} : contrôle du parseur (sans Qwen)", flush=True)
            continue

        concept_id, question_id = case["concept_id"], case["question_id"]
        try:
            concept = load_concept(concept_id)
        except FileNotFoundError:
            concept = concept_id  # L'API produira le refus contrôlé du bloc B.
        found = isinstance(concept, dict) and any(
            question["id"] == question_id for question in concept["questions"]
        )
        source = "qwen_baseline" if found else "garde_inference"
        history = []  # Isolation indispensable entre N1, N2, N3, etc.
        for tour, turn in enumerate(case["turns"], start=1):
            user = {"role": "user", "content": turn["user_input"]}
            result = generate_response(
                messages=history + [user], concept=concept, question=question_id,
                model_version="baseline", prompt_version=prompt_version,
                max_history_turns=max_history_turns,
                max_new_tokens=max_new_tokens,
            )
            predictions.append({"id": case["id"], "tour": tour,
                                "source": source, **result})
            history.extend([user, {"role": "assistant",
                                   "content": result["raw_response"]}])
            print(f"{case['id']}/{tour} : source={source}, "
                  f"verdict={result['verdict']}, "
                  f"format_valide={result['valid_format']}", flush=True)
    return predictions


def save_results(cases: list[dict], predictions: list[dict], output_dir: Path,
                 *, prompt_version: str, max_history_turns: int,
                 max_new_tokens: int) -> dict:
    """Sauvegarde les sorties réelles, le détail des scores et les métadonnées."""
    details, metrics = score_suite(cases, predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    prefix = output_dir / f"baseline_{prompt_version}_{stamp}"
    predictions_path = Path(f"{prefix}_predictions.jsonl")
    details_path = Path(f"{prefix}_details.csv")
    metrics_path = Path(f"{prefix}_metrics.json")

    with predictions_path.open("w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    with details_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    report = {
        "configuration": {
            "model_version": "baseline",
            "base_model": "Qwen/Qwen2.5-3B-Instruct",
            "prompt_version": prompt_version,
            "max_history_turns": max_history_turns,
            "max_new_tokens": max_new_tokens,
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "portee": "pilote T6.1, pas évaluation générale",
        },
        "metriques": metrics,
    }
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    return {"predictions": predictions_path, "details": details_path,
            "metrics": metrics_path, "report": report}


def main() -> None:
    """Prépare les cas, lance l'inférence et sauvegarde les résultats."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cases", type=Path, default=ROOT / "evaluation/cases_v0.1.jsonl")
    cli.add_argument("--output-dir", type=Path, default=ROOT / "evaluation/results")
    cli.add_argument("--prompt", choices=("criteria", "fewshot"), default="fewshot")
    cli.add_argument("--only", metavar="ID", help="Exécuter un seul scénario (ex. N2)")
    cli.add_argument("--max-history-turns", type=int, default=3)
    cli.add_argument("--max-new-tokens", type=int, default=160)
    cli.add_argument("--dry-run", action="store_true", help="Vérifier les cas, sans Qwen")
    args = cli.parse_args()

    cases = load_jsonl(args.cases)
    if args.only:
        cases = [case for case in cases if case["id"] == args.only]
        if not cases:
            cli.error(f"Scénario inconnu : {args.only}")
    planned = planned_tests(cases)
    print(f"Préparation OK : {len(cases)} scénarios, {len(planned)} observations")
    if args.dry_run:
        for case in cases:
            print(f"  {case['id']} : {case['type']}")
        print("Aucun modèle chargé ; aucun score mesuré.")
        return

    predictions = run_cases(cases, prompt_version=args.prompt,
                            max_history_turns=args.max_history_turns,
                            max_new_tokens=args.max_new_tokens)
    saved = save_results(cases, predictions, args.output_dir,
                         prompt_version=args.prompt,
                         max_history_turns=args.max_history_turns,
                         max_new_tokens=args.max_new_tokens)
    print("\nMétriques :")
    print(json.dumps(saved["report"]["metriques"], indent=2, ensure_ascii=False))
    for name in ("predictions", "details", "metrics"):
        print(f"{name} : {saved[name]}")
    print("Refus N5/X3 : relecture humaine obligatoire ; "
          "le refus formel ne prouve pas sa pertinence.")
    if args.only:
        print("ATTENTION : sous-ensemble exploratoire, pas score global T6.1.")


if __name__ == "__main__":
    main()
