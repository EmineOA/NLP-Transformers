"""T6.2 / T6.3 : 20 requêtes gelées sur baseline ou futur LoRA.

Exemples depuis la racine :
  python evaluation/run_final.py --prepare
  python evaluation/run_final.py --dry-run
  python evaluation/run_final.py --only T11
  python evaluation/run_final.py --model baseline
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.final_evaluation import (
    CASES, FOLDER, build_scenarios, create_frozen_cases,
    load_frozen_cases, file_sha256,
)
from evaluation.score import planned_tests, score_suite


def evaluate(scenarios: list[dict], *, model_version: str = "baseline",
             prompt_version: str = "fewshot", max_history_turns: int = 3,
             max_new_tokens: int = 160, only: str | None = None,
             components=None) -> tuple[list[dict], bool]:
    """Rejoue les scénarios en conservant l'historique seulement au sein d'une paire.

    `only` exécute le scénario contenant le test choisi (deux tours pour une paire)
    et ne produit PAS de score global. Le modèle reçoit uniquement user_input.
    """
    from src.agents.inference import generate_response

    partial = only is not None
    if only:
        scenarios = [s for s in scenarios if any(t["test_id"] == only for t in s["turns"])]
        if not scenarios:
            raise ValueError(f"Test inconnu : {only}")
    if components is None:
        if model_version == "lora":
            raise NotImplementedError(
                "Brancher un chargeur LoRA/QLoRA du bloc C dans evaluation/run_final.py ; "
                "ne jamais utiliser le modèle de base comme adaptateur."
            )
        from src.llm.model import load_model
        components = load_model()

    predictions = []
    for case in scenarios:
        history = []
        for tour, turn in enumerate(case["turns"], start=1):
            user = {"role": "user", "content": turn["user_input"]}
            # NOTE : ne PAS utiliser turn['expected'] dans l'appel au LLM.
            result = generate_response(
                history + [user], case["concept_id"], case["question_id"],
                model_version=model_version, prompt_version=prompt_version,
                max_history_turns=max_history_turns, max_new_tokens=max_new_tokens,
                components=components,
            )
            predictions.append({
                "id": case["id"], "tour": tour, "test_id": turn["test_id"],
                "source": "qwen_" + model_version, **result,
            })
            history.extend([user, {"role": "assistant", "content": result["raw_response"]}])
            print(f"{turn['test_id']} : obtenu={result['verdict']} valide={result['valid_format']}", flush=True)
    return predictions, partial


def save_run(scenarios: list[dict], predictions: list[dict], output_dir: Path,
             *, model_version: str, prompt_version: str,
             max_history_turns: int, max_new_tokens: int, partial: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    prefix = output_dir / f"{model_version}_{prompt_version}_{stamp}"
    pred_path = Path(str(prefix) + "_predictions.jsonl")
    pred_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8"
    )
    meta = {
        "version": "v1.0", "suite_sha256": file_sha256(CASES),
        "model_version": model_version, "prompt_version": prompt_version,
        "max_history_turns": max_history_turns, "max_new_tokens": max_new_tokens,
        "generation": "greedy (do_sample=False)", "partial": partial,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_observations": len(predictions),
    }
    out = {"predictions": pred_path}
    if not partial:
        details, metrics = score_suite(scenarios, predictions)
        test_ids = {(s["id"], n): turn["test_id"]
                    for s in scenarios for n, turn in enumerate(s["turns"], start=1)}
        for detail in details:
            detail["test_id"] = test_ids[(detail["id"], detail["tour"])]
        if len(predictions) != 20:
            raise ValueError("20 observations attendues pour le score final")
        details_path = Path(str(prefix) + "_details.csv")
        with details_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(details[0]))
            writer.writeheader()
            writer.writerows(details)
        meta["metriques"] = metrics
        out["details"] = details_path
    metrics_path = Path(str(prefix) + "_metrics.json")
    metrics_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    out["metrics"] = metrics_path
    return out


def create_human_template(scenarios: list[dict], path: Path) -> Path:
    """Prépare la grille de Massinissa pour les 20 sorties baseline et LoRA."""
    if path.exists():
        raise FileExistsError("Grille déjà existante : conserver la notation humaine")
    fields = [
        "test_id", "scenario_id", "tour", "modele",
        "pertinence_0_2", "role_format_0_2", "historique_0_2",
        "hors_perimetre_0_2", "absence_inventions_0_2",
        "refus_effectif_oui_non_na", "commentaires", "evaluateur",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model in ("baseline", "lora"):
            for scenario in scenarios:
                for tour, item in enumerate(scenario["turns"], start=1):
                    writer.writerow({"test_id": item["test_id"],
                                     "scenario_id": scenario["id"],
                                     "tour": tour, "modele": model})
    return path


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--prepare", action="store_true", help="Crée cases.csv et lock.json une seule fois")
    cli.add_argument("--dry-run", action="store_true", help="Vérification structurelle sans charger Qwen")
    cli.add_argument("--only", metavar="Txx", help="Un test, avec le tour antérieur s'il est apparié")
    cli.add_argument("--model", choices=("baseline", "lora"), default="baseline")
    cli.add_argument("--prompt", choices=("criteria", "fewshot"), default="fewshot")
    cli.add_argument("--max-history-turns", type=int, default=3)
    cli.add_argument("--max-new-tokens", type=int, default=160)
    cli.add_argument("--output-dir", type=Path, default=FOLDER / "results")
    args = cli.parse_args()

    if args.prepare:
        print(json.dumps(create_frozen_cases(), indent=2, ensure_ascii=False))
        rows, selection = load_frozen_cases()
        scenarios = build_scenarios(rows, selection)
        sheet = create_human_template(scenarios, FOLDER / "grille_humaine.csv")
        print("Grille humaine vierge :", sheet)
        print("Important : relire les verdicts/feedbacks puis committer cases.csv + lock.json ensemble.")
        return

    rows, selection = load_frozen_cases()
    scenarios = build_scenarios(rows, selection)
    print(f"Suite {selection['version']} : 20 requêtes, {len(scenarios)} scénarios, "
          f"3 conversations de deux tours ; empreinte SHA256 {file_sha256(CASES)}")
    if args.dry_run:
        for s in scenarios:
            print(f"{s['id']} : {', '.join(t['test_id'] for t in s['turns'])}")
        print("OK : tests gelés valides. Aucun modèle chargé, aucun score mesuré.")
        return

    try:
        predictions, partial = evaluate(
            scenarios, model_version=args.model, prompt_version=args.prompt,
            max_history_turns=args.max_history_turns,
            max_new_tokens=args.max_new_tokens, only=args.only,
        )
    except NotImplementedError as exc:
        raise SystemExit(f"Modèle {args.model} indisponible dans src/agents/inference.py : {exc}") from exc
    saved = save_run(
        scenarios, predictions, args.output_dir, model_version=args.model,
        prompt_version=args.prompt, max_history_turns=args.max_history_turns,
        max_new_tokens=args.max_new_tokens, partial=partial,
    )
    print("Sorties enregistrées :", saved)
    if partial:
        print("SOUS-ENSEMBLE : ne pas interpréter les résultats comme score final.")
    else:
        print("Les notes humaines et les refus effectifs restent à relire ; le score automatique ne les remplace pas.")


if __name__ == "__main__":
    main()
