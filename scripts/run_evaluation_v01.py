"""Génère les prédictions réelles sur les cas du bloc D, sans lire 'expected'.

Le dossier evaluation/ doit être intégré depuis la branche evaluation.
Ce script ne calcule PAS de score et n'invente pas 'refus_effectif'.
"""
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.baseline import parse_pedagogical_response
from src.agents.inference import generate_response
from src.llm.model import load_model


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--cases", default="evaluation/cases_v0.1.jsonl")
    cli.add_argument("--output", default="evaluation/predictions_v0.1_baseline.jsonl")
    cli.add_argument("--prompt", choices=("criteria", "fewshot"), default="fewshot")
    args = cli.parse_args()
    cases = [json.loads(s) for s in Path(args.cases).read_text(encoding="utf-8").splitlines() if s.strip()]

    # Modèle chargé au premier cas de conversation seulement, puis réutilisé.
    llm = None
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fp:
        for case in cases:
            if case["type"] == "parser":
                out = parse_pedagogical_response(case["raw_response"])
                fp.write(json.dumps({"id": case["id"], "tour": 1, **out}, ensure_ascii=False) + "\n")
                continue
            # Scénarios isolés : pas d'historique commun entre N1 et N2, etc.
            history = []
            for tour, step in enumerate(case["turns"], start=1):
                # NE PAS passer step['expected'] au LLM.
                if llm is None:
                    llm = load_model()
                user = {"role": "user", "content": step["user_input"]}
                out = generate_response(
                    history + [user], case["concept_id"], case["question_id"],
                    prompt_version=args.prompt, components=llm,
                )
                fp.write(json.dumps({"id": case["id"], "tour": tour, **out}, ensure_ascii=False) + "\n")
                history += [user, {"role": "assistant", "content": out["raw_response"]}]
                print(case["id"], tour, out["verdict"], out["valid_format"], flush=True)
    print("Prédictions sauvegardées :", target)
    print("ATTENTION : relire les refus et noter 'refus_effectif' manuellement.")


if __name__ == "__main__":
    main()
