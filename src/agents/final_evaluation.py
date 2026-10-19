"""Protocole figé T6.2 (20 requêtes) au format CSV de birdyy667.

Les verdicts attendus servent EXCLUSIVEMENT au scoring. L'appel au moteur
ne reçoit que le texte utilisateur (avec le contexte historique) et les
réponses du moteur lors de dialogues multi-tours.

Aucune dépendance à torch : préparation et vérification exécutables à froid.
"""
from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "evaluation" / "final_v1.0"
SELECTION = FOLDER / "selection.json"
CASES = FOLDER / "cases.csv"
LOCK = FOLDER / "lock.json"
SOURCE = ROOT / "data" / "dialogue" / "classification" / "dataset_source.csv"

LABELS = {"PARFAIT", "PARTIEL", "INCORRECT", "HORS_PERIMETRE"}
HOLDOUT_QUESTIONS = {
    "bateau-de-thesee_q3",
    "doute-de-descartes_q1",
    "infinis-plus-grands-que-l-infini_q2",
}
REQUIRED_COLUMNS = (
    "id", "concept_id", "question_id", "enonce", "essai_question",
    "labels_historique", "historique", "reponse_utilisateur", "label",
    "feedback", "cas", "style", "split", "generateur",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        fields = reader.fieldnames or []
        if fields != list(REQUIRED_COLUMNS):
            raise ValueError(f"Colonnes incompatibles avec le format Jalis : {fields}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("CSV mal formé : séparateurs ou colonnes manquantes")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("IDs du CSV dupliqués")
    return fields, rows


def load_selection(path: Path = SELECTION) -> dict:
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("version") != "v1.0":
        raise ValueError("Version de la sélection inattendue")
    tests = selection.get("tests", [])
    if len(tests) != 20 or [t["id"] for t in tests] != [f"T{i:02d}" for i in range(1, 21)]:
        raise ValueError("20 tests T01..T20 exactement sont obligatoires")
    if len(selection.get("dialogues", [])) != 3:
        raise ValueError("Trois conversations de deux tours sont obligatoires")
    return selection


def _historique_avant_essai(row: dict, student_response: str, expected_label: str) -> tuple[str, str]:
    """Extrait le contexte précédant l'essai 1, en vérifiant son texte/label.

    Ne fabrique pas de nouvelles réponses utilisateur : reproduit une ligne
    déjà présente dans l'historique du CSV de Jalis.
    """
    if row["essai_question"] != "2":
        raise ValueError(f"Source attendue à l'essai 2 : {row['id']}")
    marker = f"[{row['question_id']} – essai 1] "
    lines = row["historique"].splitlines()
    if not lines or not lines[-1].startswith(marker):
        raise ValueError(f"Historique sans essai 1 final : {row['id']}")
    tail = lines[-1][len(marker):]
    tag = "[HORS_PERIMETRE]" if expected_label == "HORS_PERIMETRE" else f"[VERDICT: {expected_label}]"
    if tail != f"{student_response} → {tag}":
        raise ValueError(f"Réponse précédente différente du texte fixé : {row['id']}")
    previous_labels = [x.strip() for x in row["labels_historique"].split(" | ") if x.strip()]
    if not previous_labels or previous_labels[-1] != f"{row['question_id']} essai 1:{expected_label}":
        raise ValueError(f"Labels de l'historique incohérents : {row['id']}")
    return "\n".join(lines[:-1]), " | ".join(previous_labels[:-1])


def _synthetic_row(source: dict, test: dict) -> dict:
    history, labels = _historique_avant_essai(source, test["student_response"], test["label"])
    row = dict(source)
    row.update({
        "id": test["id"],
        "essai_question": "1",
        "labels_historique": labels,
        "historique": history,
        "reponse_utilisateur": test["student_response"],
        "label": test["label"],
        "feedback": test["feedback"],
        "cas": "essai précédent reconstitué depuis l'historique",
        "style": "naturel",
        "generateur": "reconstruction_manuelle",
    })
    return row


def create_frozen_cases(source: Path = SOURCE, selection_path: Path = SELECTION,
                        output: Path = CASES, lock_path: Path = LOCK) -> dict:
    """Matérialise le CSV figé UNE FOIS. Refuse d'écraser un gel existant.

    La SHA Git du CSV source est verrouillée sur le commit de référence.
    La sortie et son lock sont ensuite à committer ensemble sur main.
    """
    if output.exists() or lock_path.exists():
        raise FileExistsError("Suite déjà figée : ne pas régénérer ni modifier après les scores")
    selection = load_selection(selection_path)
    if git_blob_sha(source) != selection["source_blob_sha"]:
        raise ValueError("Le dataset Jalis a changé par rapport au commit gelé")
    fields, rows = read_csv(source)
    by_id = {row["id"]: row for row in rows}
    frozen = []
    for test in selection["tests"]:
        if test["source_id"] not in by_id:
            raise ValueError(f"Source manquante : {test['source_id']}")
        row = by_id[test["source_id"]]
        if row["split"] != "test" or row["question_id"] not in HOLDOUT_QUESTIONS:
            raise ValueError(f"Fuite train/val dans la sélection : {test['id']}")
        if test.get("synthetic_previous"):
            row = _synthetic_row(row, test)
        else:
            row = dict(row, id=test["id"])
        frozen.append(row)
    validate_suite(frozen, selection)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=";", lineterminator="\n")
        writer.writeheader()
        writer.writerows(frozen)
    locked = {
        "version": "v1.0",
        "selection_sha256": file_sha256(selection_path),
        "cases_sha256": file_sha256(output),
        "dataset_git_blob_sha": selection["source_blob_sha"],
        "source_commit": selection["source_commit"],
        "nombre_requetes": 20,
    }
    lock_path.write_text(json.dumps(locked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return locked


def validate_suite(rows: list[dict], selection: dict) -> None:
    if len(rows) != 20 or [row["id"] for row in rows] != [f"T{i:02d}" for i in range(1, 21)]:
        raise ValueError("La suite doit contenir exactement T01 à T20")
    if Counter(row["label"] for row in rows) != {label: 5 for label in LABELS}:
        raise ValueError("Il faut cinq exemples de chaque verdict")
    if {row["question_id"] for row in rows} != HOLDOUT_QUESTIONS:
        raise ValueError("Questions test incohérentes avec le split Jalis")
    for row in rows:
        if row["split"] != "test" or row["label"] not in LABELS:
            raise ValueError(f"Ligne non test ou label invalide : {row['id']}")
        if row["essai_question"] not in {"1", "2", "3"}:
            raise ValueError(f"Numéro d'essai invalide : {row['id']}")
        if not row["reponse_utilisateur"].strip() or not row["feedback"].strip():
            raise ValueError(f"Texte de test vide : {row['id']}")
    used = set()
    index = {row["id"]: row for row in rows}
    for dialogue in selection["dialogues"]:
        parts = dialogue["turns"]
        if len(parts) != 2 or any(t in used or t not in index for t in parts):
            raise ValueError("Dialogue invalide / ID réutilisé")
        one, two = [index[x] for x in parts]
        if one["question_id"] != two["question_id"] or one["essai_question"] != "1" or two["essai_question"] != "2":
            raise ValueError("Un dialogue doit correspondre aux essais 1 puis 2 d'une même question")
        used.update(parts)
    if len(used) != 6:
        raise ValueError("Trois paires de deux tours attendues")


def load_frozen_cases(cases_path: Path = CASES, selection_path: Path = SELECTION,
                      lock_path: Path = LOCK) -> tuple[list[dict], dict]:
    """Vérifie les SHA pour interdire les modifications silencieuses après gel."""
    selection = load_selection(selection_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("version") != selection["version"]:
        raise ValueError("Versions de suite différentes")
    if file_sha256(selection_path) != lock["selection_sha256"] or file_sha256(cases_path) != lock["cases_sha256"]:
        raise ValueError("Suite modifiée après gel : générer une NOUVELLE version")
    _, rows = read_csv(cases_path)
    validate_suite(rows, selection)
    return rows, selection


def _expected(label: str) -> dict:
    return {"verdict": None if label == "HORS_PERIMETRE" else label,
            "hors_perimetre": label == "HORS_PERIMETRE"}


def _previous_question_context(row: dict) -> str:
    """Retire uniquement l'essai 1 de la question courante (déjà envoyé en vrai tour)."""
    marker = f"[{row['question_id']} – essai 1] "
    lines = row["historique"].splitlines()
    if not lines or not lines[-1].startswith(marker):
        raise ValueError(f"Impossible de séparer le tour précédent : {row['id']}")
    return "\n".join(lines[:-1])


def format_user_input(row: dict, *, prior_turn_in_chat: bool = False) -> str:
    """Même assemblage du message user que export_jsonl.build_messages de Jalis.

    Exception pour le tour 2 d'une paire : l'essai 1 est un VRAI couple
    user/assistant dans l'historique chat, donc retiré du résumé pour ne pas le
    présenter deux fois ou injecter son verdict de référence.
    """
    from src.agents.concepts import load_concept, load_question
    from data.dialogue.classification.export_jsonl import build_messages

    concept = load_concept(row["concept_id"])
    question = load_question(concept, row["question_id"])
    payload = dict(row)
    if prior_turn_in_chat:
        payload["historique"] = _previous_question_context(row)
    messages = build_messages(payload, concept, question)
    return messages[1]["content"]


def build_scenarios(rows: list[dict], selection: dict) -> list[dict]:
    """17 scénarios, 20 requêtes : 3 paires + 14 singletons.

    Les scénarios utilisent la structure que sait déjà noter evaluation.score.
    Le verdict de chaque tour n'est PAS joint aux messages du modèle.
    """
    validate_suite(rows, selection)
    by_id = {r["id"]: r for r in rows}
    pairs = {d["turns"][0]: d for d in selection["dialogues"]}
    seconds = {d["turns"][1] for d in selection["dialogues"]}
    scenarios = []
    for row in rows:
        ident = row["id"]
        if ident in seconds:
            continue
        if ident in pairs:
            dlg = pairs[ident]
            a, b = (by_id[t] for t in dlg["turns"])
            cases = [(a, False), (b, True)]
            scenario_id = dlg["id"]
        else:
            cases = [(row, False)]
            scenario_id = ident
        scenarios.append({
            "id": scenario_id,
            "type": "conversation",
            "concept_id": row["concept_id"],
            "question_id": row["question_id"],
            "turns": [{
                "test_id": r["id"],
                "user_input": format_user_input(r, prior_turn_in_chat=previous_chat),
                "expected": _expected(r["label"]),
            } for r, previous_chat in cases],
        })
    if len(scenarios) != 17 or sum(len(s["turns"]) for s in scenarios) != 20:
        raise AssertionError("Le nombre de scénarios/requêtes est incorrect")
    return scenarios
