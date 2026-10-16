"""Prépare le lot à envoyer au générateur : les lignes d'une question attribuées à un modèle.

Affiche le bloc {{LIGNES_JSON}} du prompt (colonnes id, essai_question, labels_historique,
label, cas, style), avec le chemin de la fiche à mettre dans {{FICHE_MARKDOWN}} et l'énoncé.

Usage : python data/dialogue/classification/preparer_lot.py claude-opus-5.5 monty-hall_q1
"""
import csv
import glob
import json
import os
import sys

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(os.path.dirname(os.path.dirname(ICI)))
COLONNES = ["id", "essai_question", "labels_historique", "label", "cas", "style"]


def main():
    generateur, question_id = sys.argv[1], sys.argv[2]
    with open(os.path.join(ICI, "dataset_source.csv"), encoding="utf-8-sig", newline="") as f:
        lignes = [l for l in csv.DictReader(f, delimiter=";")
                  if l["question_id"] == question_id and l["generateur"] == generateur]
    if not lignes:
        raise SystemExit(f"Aucune ligne pour {generateur} / {question_id}")
    concept_id = lignes[0]["concept_id"]
    fiche = glob.glob(os.path.join(RACINE, "data", "concepts", "markdown", "*", f"{concept_id}.md"))[0]
    print("FICHE_MARKDOWN :", os.path.relpath(fiche, RACINE))
    print("QUESTION_ID    :", question_id)
    print("ENONCE         :", lignes[0]["enonce"])
    print(f"LIGNES_JSON    : {len(lignes)} lignes")
    print(json.dumps([{k: l[k] for k in COLONNES} for l in lignes], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
