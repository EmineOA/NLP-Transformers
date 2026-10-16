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
    context = []
    if row["historique"]:
        context.append("HISTORIQUE RÉSUMÉ (contexte, pas la réponse à évaluer) :\n" + row["historique"])
    if attempt > 1:
        context.append("INDICES DÉJÀ FOURNIS (selon le numéro d'essai) :\n" + "\n".join(question["indices"][:attempt - 1]))
    context.append("RÉPONSE ACTUELLE À ÉVALUER :\n" + row["reponse_utilisateur"])
    return [
        {"role": "system", "content": build_pedagogical_prompt(concept, question, version=prompt_version)},
        {"role": "user", "content": "\n\n".join(context)},
        {"role": "assistant", "content": TAGS[row["label"]] + "\n" + row["feedback"]},
    ]
