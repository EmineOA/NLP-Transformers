"""Export d'exemples côte à côte « modèle de base » contre « LoRA » (bloc C).

Les chiffres de evaluate_verdicts disent combien de verdicts sont justes, mais pas à quoi ressemblent
les réponses. Ce fichier choisit quelques exemples variés de val (select_cases), fait répondre le
LoRA puis le modèle de base sur chacun, et écrit le tout dans un petit JSON pour un notebook de lecture.
Il lit les deux fichiers JSON écrits par evaluate_verdicts (clé "rows") pour savoir où chaque modèle se trompe.

Commande à lancer sur Colab (GPU A100), depuis la racine du dépôt :
!python -m src.training.export_examples --adapter {RUN_DIR}/best --eval-base {EVAL_DIR}/eval_v1.0_base.json --eval-lora {EVAL_DIR}/eval_v1.0_lora.json --output {EVAL_DIR}/exemples_v1.0.json
"""

import argparse
import csv
import json
from pathlib import Path

from src.training.dataset import DATA_DIR, load_split
from src.training.evaluate_verdicts import label_of
from src.training.model import load_model_with_adapter
from src.training.train import generate_answer

# Au-delà, la réponse de l'élève rend le JSON pénible à lire dans le notebook.
MAX_STUDENT_CHARS = 700


def truncate_text(text: str, max_chars: int = MAX_STUDENT_CHARS) -> str:
    """Coupe un texte trop long et ajoute « … » à la fin.

    Paramètres :
        text : texte à raccourcir.
        max_chars : nombre maximal de caractères gardés avant le « … ».

    Retour :
        le texte tel quel s'il est assez court, sinon ses max_chars premiers caractères suivis de « … ».

    Exemple :
        truncate_text("abcdef", 3)  # "abc…"
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def reason_of(base_ok: bool, lora_ok: bool) -> str:
    """Décrit en quelques mots qui a juste et qui a faux.

    Paramètres :
        base_ok : le modèle de base a le bon verdict.
        lora_ok : le LoRA a le bon verdict.

    Retour :
        une étiquette de raison en français.

    Exemple :
        reason_of(False, True)  # "LoRA juste, base fausse"
    """
    if base_ok and lora_ok:
        return "les deux justes"
    if lora_ok:
        return "LoRA juste, base fausse"
    if base_ok:
        return "LoRA se trompe (base juste)"
    return "les deux se trompent"


def pick_varied(candidates: list[dict], count: int, chosen: list[dict], question_ids: dict, max_per_question: int | None = None) -> list[dict]:
    """Prend `count` candidats en variant les verdicts attendus et les questions.

    Paramètres :
        candidates : lignes possibles (clés "index" et "expected").
        count : nombre de candidats à prendre.
        chosen : lignes déjà choisies (elles servent à ne pas répéter verdict et question).
        question_ids : {index val : question_id}, peut être vide.
        max_per_question : nombre maximal de cas par question (None : pas de limite).

    Retour :
        les lignes prises, dans l'ordre de choix (moins que `count` s'il n'y en a pas assez).

    Exemple :
        pick_varied([{"index": 3, "expected": "PARTIEL"}], 1, [], {})  # [{"index": 3, ...}]
    """
    taken = []
    for _ in range(count):
        already = chosen + taken
        used_indices = {row["index"] for row in already}
        remaining = [row for row in candidates if row["index"] not in used_indices]
        if max_per_question is not None:
            # une question qui a déjà son quota est écartée : sans cela, val (2 questions) ne serait pas équilibré
            remaining = [row for row in remaining if sum(1 for other in already if question_ids.get(other["index"]) == question_ids.get(row["index"])) < max_per_question]
        if not remaining:
            break
        used_questions = {question_ids.get(row["index"]) for row in already} - {None}

        # clé : d'abord un verdict encore peu vu, puis une question inédite, puis l'index le plus petit
        # (l'index départage les égalités, donc le résultat est identique à chaque appel)
        def sort_key(row):
            same_verdict = sum(1 for other in already if other["expected"] == row["expected"])
            same_question = question_ids.get(row["index"]) in used_questions
            return (same_verdict, same_question, row["index"])

        taken.append(min(remaining, key=sort_key))
    return taken


def select_cases(rows_base: list[dict], rows_lora: list[dict], number: int, question_ids: dict | None = None) -> list[tuple[int, str]]:
    """Choisit `number` exemples variés, en privilégiant les plus instructifs.

    Ordre de priorité : 2 « LoRA juste, base fausse » ; 1 refus détecté par le LoRA ;
    1 où le LoRA se trompe ; 1 où les deux ont juste sur PARTIEL ou INCORRECT ; puis le reste.
    Si question_ids est fourni, une question a au plus ceil(number / 2) cas ; une raison qui manque
    pour une question est remplacée par la raison suivante.

    Paramètres :
        rows_base : lignes {"index", "expected", "predicted", "correct"} du modèle de base.
        rows_lora : mêmes lignes pour le LoRA.
        number : nombre d'exemples voulus.
        question_ids : {index val : question_id} pour éviter de répéter une question (optionnel).

    Retour :
        liste de (index val, raison), sans doublon, dans l'ordre de choix.

    Exemple :
        select_cases(rows_base, rows_lora, 6)  # [(12, "LoRA juste, base fausse"), (40, "refus détecté"), ...]
    """
    question_ids = question_ids or {}
    max_per_question = -(-number // 2) if question_ids else None  # arrondi vers le haut
    base_by_index = {row["index"]: row for row in rows_base}
    # seuls les index présents dans les deux fichiers sont comparables ; les lignes du LoRA portent l'attendu
    pairs = [(base_by_index[row["index"]], row) for row in sorted(rows_lora, key=lambda r: r["index"]) if row["index"] in base_by_index]

    lora_wins = [lora for base, lora in pairs if lora["correct"] and not base["correct"]]
    refusals = [lora for _, lora in pairs if lora["correct"] and lora["expected"] == "HORS_PERIMETRE"]
    lora_errors = [lora for _, lora in pairs if not lora["correct"]]
    both_ok = [lora for base, lora in pairs if lora["correct"] and base["correct"] and lora["expected"] in ("PARTIEL", "INCORRECT")]

    chosen = []
    reasons = {}
    steps = [(lora_wins, 2, "LoRA juste, base fausse"), (refusals, 1, "refus détecté"),
             (lora_errors, 1, "LoRA se trompe"), (both_ok, 1, "les deux justes")]
    for candidates, count, reason in steps:
        if len(chosen) >= number:
            break
        for row in pick_varied(candidates, min(count, number - len(chosen)), chosen, question_ids, max_per_question):
            chosen.append(row)
            reasons[row["index"]] = reason

    # on complète avec tout ce qui reste ; la raison est alors calculée d'après les résultats réels
    if len(chosen) < number:
        base_ok = {base["index"]: base["correct"] for base, _ in pairs}
        remaining = [lora for _, lora in pairs]
        for limit in (max_per_question, None):
            # 2e passage sans quota : s'il n'y a qu'une seule question, on préfère remplir que rester court
            for row in pick_varied(remaining, number - len(chosen), chosen, question_ids, limit):
                chosen.append(row)
                reasons[row["index"]] = reason_of(base_ok[row["index"]], row["correct"])
    return [(row["index"], reasons[row["index"]]) for row in chosen]


def load_val_question_ids() -> list[str | None]:
    """Lit la colonne question_id de dataset_source.csv pour les lignes du split val.

    Retour :
        un question_id par conversation de val, dans le même ordre que load_split("val").

    Exemple :
        load_val_question_ids()[0]  # "bateau-de-thesee_q1"
    """
    # même lecture que load_split (BOM Excel, séparateur ";") : l'ordre du CSV est celui du JSONL
    with open(DATA_DIR / "dataset_source.csv", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file, delimiter=";"))
    return [row.get("question_id") for row in rows if row["split"] == "val"]


def last_content(conversation: list[dict], role: str) -> str:
    """Retourne le texte du dernier message d'un rôle donné ("user" ou "assistant").

    Exemple :
        last_content([{"role": "user", "content": "a"}], "user")  # "a"
    """
    return [message["content"] for message in conversation if message["role"] == role][-1]


def build_cases(model, tokenizer, selected: list[tuple[int, str]], val: list[list[dict]], question_ids: list) -> list[dict]:
    """Fait répondre le LoRA puis le modèle de base sur chaque exemple choisi.

    Paramètres :
        model : modèle 4 bits avec l'adaptateur LoRA (load_model_with_adapter).
        tokenizer : tokenizer du modèle.
        selected : sortie de select_cases.
        val : conversations de load_split("val").
        question_ids : sortie de load_val_question_ids.

    Retour :
        une liste de cas (format du JSON de sortie).

    Exemple :
        build_cases(model, tokenizer, [(3, "refus détecté")], val, question_ids)[0]["lora"]["verdict"]
    """
    cases = []
    for val_index, reason in selected:
        conversation = val[val_index]
        prompt_messages = conversation[:-1]  # sans la réponse attendue, sinon le modèle la recopierait
        lora_text = generate_answer(model, tokenizer, prompt_messages)
        # adaptateur coupé = modèle de base, sans le recharger (même méthode que le notebook §7)
        with model.disable_adapter():
            base_text = generate_answer(model, tokenizer, prompt_messages)

        expected_text = last_content(conversation, "assistant")
        cases.append({
            "val_index": val_index,
            "question_id": question_ids[val_index] if val_index < len(question_ids) else None,
            "raison": reason,
            "verdict_attendu": label_of(expected_text),
            "reponse_eleve": truncate_text(last_content(conversation, "user")),
            "reponse_attendue": expected_text,
            "base": {"texte": base_text, "verdict": label_of(base_text)},
            "lora": {"texte": lora_text, "verdict": label_of(lora_text)},
        })
        print(f"cas val n°{val_index} ({reason}) : base {cases[-1]['base']['verdict']}, LoRA {cases[-1]['lora']['verdict']}")
    return cases


def make_row(index: int, expected: str, predicted: str) -> dict:
    """Fabrique une ligne au format de evaluate_verdicts (pour les tests)."""
    return {"index": index, "expected": expected, "predicted": predicted, "correct": expected == predicted}


def self_test() -> None:
    """Teste select_cases et truncate_text sans modèle (pas de GPU nécessaire).

    Retour :
        rien (lève AssertionError si un cas échoue).
    """
    expected = ["PARFAIT", "PARTIEL", "INCORRECT", "HORS_PERIMETRE"] * 4  # 16 exemples
    # base : toujours PARFAIT ; LoRA : toujours juste sauf l'exemple 5
    rows_base = [make_row(i, e, "PARFAIT") for i, e in enumerate(expected)]
    rows_lora = [make_row(i, e, "PARFAIT" if i == 5 else e) for i, e in enumerate(expected)]

    cases = select_cases(rows_base, rows_lora, 6)
    indices = [index for index, _ in cases]
    assert len(cases) == 6 and len(set(indices)) == 6, "6 cas sans doublon"
    assert [r for _, r in cases[:2]] == ["LoRA juste, base fausse"] * 2
    assert cases[2][1] == "refus détecté" and expected[cases[2][0]] == "HORS_PERIMETRE"
    assert cases[3] == (5, "LoRA se trompe")
    assert len({expected[index] for index in indices}) >= 3, "verdicts attendus variés"
    assert select_cases(rows_base, rows_lora, 6) == cases, "déterministe"

    # pas de cas « LoRA juste, base fausse » : la base est identique au LoRA
    same = select_cases(rows_lora, rows_lora, 6)
    assert len(same) == 6 and all(r != "LoRA juste, base fausse" for _, r in same)

    # 2 questions déséquilibrées (tous les cas intéressants sur la question A, index 0 à 15) : 3 et 3 attendus
    questions = {i: ("A" if i < 16 else "B") for i in range(24)}
    rows_base2 = [make_row(i, e, "PARFAIT") for i, e in enumerate(expected + expected[:8])]
    rows_lora2 = [make_row(i, e, "PARFAIT" if i == 5 else e) for i, e in enumerate(expected + expected[:8])]
    balanced = select_cases(rows_base2, rows_lora2, 6, questions)
    counts = [sum(1 for i, _ in balanced if questions[i] == q) for q in "AB"]
    assert counts == [3, 3], f"questions déséquilibrées : {counts}"
    assert len({i for i, _ in balanced}) == 6 and select_cases(rows_base2, rows_lora2, 6, questions) == balanced
    # une seule question : le quota est ignoré pour remplir
    assert len(select_cases(rows_base, rows_lora, 6, {i: "A" for i in range(16)})) == 6

    # moins de lignes que demandé, et rows vides
    assert len(select_cases(rows_base[:2], rows_lora[:2], 6)) == 2
    assert select_cases([], [], 6) == []

    assert truncate_text("a" * 800) == "a" * 700 + "…"
    assert truncate_text("court") == "court"
    print("self-test select_cases et truncate_text : OK")


if __name__ == "__main__":
    # Lancer depuis la racine du dépôt :
    # python -m src.training.export_examples --adapter adapters/v1.0/best --eval-base base.json --eval-lora lora.json --output exemples.json
    parser = argparse.ArgumentParser(description="Exporte des exemples côte à côte base contre LoRA.")
    parser.add_argument("--adapter", help="dossier de l'adaptateur LoRA (le dossier best)")
    parser.add_argument("--eval-base", help="JSON de evaluate_verdicts pour le modèle de base")
    parser.add_argument("--eval-lora", help="JSON de evaluate_verdicts pour le LoRA")
    parser.add_argument("--output", help="fichier JSON à écrire")
    parser.add_argument("--number", type=int, default=6, help="nombre d'exemples à exporter")
    parser.add_argument("--self-test", action="store_true", help="teste select_cases puis s'arrête, sans charger de modèle")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        raise SystemExit(0)
    if not (args.adapter and args.eval_base and args.eval_lora and args.output):
        parser.error("--adapter, --eval-base, --eval-lora et --output sont obligatoires (sauf avec --self-test)")

    rows_base = json.loads(Path(args.eval_base).read_text(encoding="utf-8"))["rows"]
    rows_lora = json.loads(Path(args.eval_lora).read_text(encoding="utf-8"))["rows"]
    val = load_split("val")
    question_ids = load_val_question_ids()

    selected = select_cases(rows_base, rows_lora, args.number, dict(enumerate(question_ids)))
    model, tokenizer = load_model_with_adapter(args.adapter)
    cases = build_cases(model, tokenizer, selected, val, question_ids)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"adapter": args.adapter, "cases": cases}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(cases)} exemples écrits dans {output_path}")
