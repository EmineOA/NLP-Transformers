"""Lecture en mode strict des fiches JSON générées par le bloc A.

Modifier les Markdown sources du bloc A, jamais les JSON dérivés ici.
"""
import json
import re
from pathlib import Path

CONCEPT_FOLDER = Path(__file__).resolve().parents[2] / "data" / "concepts" / "json"
CONCEPT_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def load_concept(concept_id: str) -> dict:
    """Charge et valide une fiche (contrat A) depuis son identifiant.

    Exemple : load_concept('monty-hall')['id'] == 'monty-hall'.
    """
    if not isinstance(concept_id, str) or not CONCEPT_ID.fullmatch(concept_id):
        raise ValueError(f"Identifiant de concept invalide : {concept_id!r}")
    path = CONCEPT_FOLDER / f"{concept_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Fiche absente du corpus : {path}")
    try:
        concept = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Fiche JSON invalide : {path.name}") from exc
    if not isinstance(concept, dict) or concept.get("id") != concept_id:
        raise ValueError(f"Fiche ou identifiant incohérent : {path.name}")
    for key in ("matiere", "concept"):
        if not isinstance(concept.get(key), str):
            raise ValueError(f"Champ {key} invalide dans {path.name}")
    sections = concept.get("explication")
    questions = concept.get("questions")
    if not isinstance(sections, list) or not sections or not isinstance(questions, list) or not questions:
        raise ValueError(f"Sections ou questions manquantes dans {path.name}")
    if any(not isinstance(s, dict) or not {"titre", "contenu"} <= s.keys() for s in sections):
        raise ValueError(f"Section de cours mal formée dans {path.name}")
    ids = set()
    for question in questions:
        if not isinstance(question, dict) or not {"id", "enonce", "apres_section", "points_cles", "indices", "niveaux"} <= question.keys():
            raise ValueError(f"Question mal formée dans {path.name}")
        if question["id"] in ids:
            raise ValueError(f"Question dupliquée : {question['id']}")
        ids.add(question["id"])
        after = question["apres_section"]
        if type(after) is not int or not 1 <= after <= len(sections):
            raise ValueError(f"Indice de section invalide dans {question['id']}")
        if not isinstance(question["points_cles"], list) or not isinstance(question["indices"], list):
            raise ValueError(f"Points clés ou indices invalides dans {question['id']}")
        levels = question["niveaux"]
        if not isinstance(levels, dict) or not {"parfait", "partiel", "incorrect", "refus"} <= levels.keys():
            raise ValueError(f"Niveaux manquants dans {question['id']}")
        for name in ("parfait", "partiel", "incorrect", "refus"):
            entry = levels[name]
            if not isinstance(entry, dict) or not isinstance(entry.get("criteres"), list) or not isinstance(entry.get("exemple"), str):
                raise ValueError(f"Niveau {name} incomplet dans {question['id']}")
    return concept


def load_question(concept: dict, question_id: str) -> dict:
    """Retrouve une question par son ID, sans changer la fiche du bloc A."""
    for question in concept["questions"]:
        if question["id"] == question_id:
            return question
    raise KeyError(f"Question inconnue dans {concept['id']} : {question_id}")
