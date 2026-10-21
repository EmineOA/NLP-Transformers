"""Export reproductible du CSV source, sans génération de nouveaux dialogues."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FOLDER = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.llm.prompts import build_pedagogical_prompt
from src.agents.concepts import load_concept, load_question
from src.agents.pedagogical_context import build_student_message

TAGS = {"PARFAIT": "[VERDICT: PARFAIT]", "PARTIEL": "[VERDICT: PARTIEL]",
        "INCORRECT": "[VERDICT: INCORRECT]", "HORS_PERIMETRE": "[HORS_PERIMETRE]"}


def build_messages(row: dict, concept: dict, question: dict, prompt_version: str = "fewshot") -> list[dict]:
    """Construit les messages système, utilisateur et assistant du contrat T4.1.

    Paramètres : ligne CSV, fiche, question et version du prompt B.
    Retour : trois messages ; seule la cible assistant contient la correction.
    Exemple : build_messages(row, concept, question)[-1]["role"] == "assistant".
    """
    if row["concept_id"] != concept["id"] or row["question_id"] != question["id"]:
        raise ValueError("La ligne ne correspond pas à la fiche et à la question.")
    if row["label"] not in TAGS or not row["reponse_utilisateur"].strip() or not row["feedback"].strip():
        raise ValueError("Label inconnu ou réponse/feedback vide.")
    attempt = int(row["essai_question"])
    if attempt not in (1, 2, 3):
        raise ValueError("L'essai doit être 1, 2 ou 3.")
    student_message = build_student_message(
        row["reponse_utilisateur"],
        history=row["historique"],
        hints_seen=question["indices"][:attempt - 1],
    )
    return [
        {"role": "system", "content": build_pedagogical_prompt(concept, question, version=prompt_version)},
        {"role": "user", "content": student_message},
        {"role": "assistant", "content": TAGS[row["label"]] + "\n" + row["feedback"]},
    ]


def export_dataset(source: Path, output: Path, prompt_version: str = "fewshot") -> int:
    """Exporte chaque ligne CSV dans un objet JSONL contenant seulement messages.

    Paramètres : chemins source/destination et version du prompt du bloc B.
    Retour : nombre d'objets écrits, dans l'ordre du CSV, sans modifier la source.
    Exemple : export_dataset(FOLDER / 'dataset_source.csv', output) == 1003.
    """
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or output.suffix != ".jsonl":
        raise ValueError("La destination doit être un fichier .jsonl distinct du CSV.")
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    if not rows:
        raise ValueError("Le CSV source est vide.")
    seen, concepts, records = set(), {}, []
    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"Identifiant en doublon : {row['id']}")
        seen.add(row["id"])
        if row["concept_id"] not in concepts:
            concepts[row["concept_id"]] = load_concept(row["concept_id"])
        concept = concepts[row["concept_id"]]
        question = load_question(concept, row["question_id"])
        records.append({"messages": build_messages(row, concept, question, prompt_version)})
    # Valider tout le lot avant d'écrire évite de livrer un export partiel.
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")
    return len(records)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", type=Path, default=FOLDER / "dataset_source.csv")
    cli.add_argument("--output", type=Path, default=FOLDER / "dataset_messages.jsonl")
    cli.add_argument("--prompt-version", choices=("criteria", "fewshot"), default="fewshot")
    args = cli.parse_args()
    count = export_dataset(args.source, args.output, args.prompt_version)
    print(f"{count} exemples exportés : {args.output}")
