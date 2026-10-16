"""Comparaison des prompts 'criteria' et 'fewshot' sur les MÊMES cas de développement.

ATTENTION : les attentes du fichier dev doivent être validées par le bloc A.
Ne jamais utiliser le jeu final de 20 requêtes pour choisir un prompt.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.concepts import load_concept
from src.agents.inference import generate_response
from src.llm.model import load_model


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--cases", default="experiments/dev_cases_bloc_b.jsonl")
    cli.add_argument("--limit", type=int, default=0)
    args = cli.parse_args()

    cases = [json.loads(x) for x in Path(args.cases).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError("Aucun cas de développement.")

    # Chargement unique du modèle, important sur une carte à mémoire limitée.
    llm = load_model()
    all_rows = []
    for prompt_version in ("criteria", "fewshot"):
        for case in cases:
            concept = load_concept(case["concept_id"])
            question = next(q for q in concept["questions"] if q["id"] == case["question_id"])
            out = generate_response(
                [{"role": "user", "content": case["user_input"]}],
                concept, question, prompt_version=prompt_version, components=llm,
            )
            all_rows.append({
                "id": case["id"], "prompt_version": prompt_version,
                "concept_id": case["concept_id"], "question_id": case["question_id"],
                "user_input": case["user_input"], "expected_dev": case["expected"],
                "verdict": out["verdict"], "correct_dev": out["verdict"] == case["expected"] and out["valid_format"],
                "valid_format": out["valid_format"], "feedback": out["feedback"],
                "raw_response": out["raw_response"],
            })
            print(f"{prompt_version:<8} {case['id']:<17} attendu={case['expected']} obtenu={out['verdict']} format={out['valid_format']}", flush=True)

    dest = Path("experiments/results")
    dest.mkdir(parents=True, exist_ok=True)
    output = dest / "comparaison_prompts_bloc_b.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {}
    for version in ("criteria", "fewshot"):
        subset = [r for r in all_rows if r["prompt_version"] == version]
        summary[version] = {
            "exactitude_dev": round(sum(r["correct_dev"] for r in subset) / len(subset), 4),
            "format_rate": round(sum(r["valid_format"] for r in subset) / len(subset), 4),
            "n": len(subset),
        }
    winner = max(summary, key=lambda p: (summary[p]["exactitude_dev"], summary[p]["format_rate"]))
    # Sauvegarde des réponses de la meilleure config sur CE jeu de développement.
    reference = dest / "baseline_reference_dev.jsonl"
    with reference.open("w", encoding="utf-8") as fp:
        for row in all_rows:
            if row["prompt_version"] == winner:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    (dest / "comparaison_prompts_resume.json").write_text(
        json.dumps({"versions": summary, "meilleure_sur_developpement": winner,
                    "attention": "Les labels de développement doivent être validés par le bloc A; ne constitue pas le test final."}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Meilleur sur développement :", winner)
    print("CSV :", output, "| Référence :", reference)


if __name__ == "__main__":
    main()
