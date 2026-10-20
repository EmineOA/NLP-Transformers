"""Mesure de la précision du verdict du tuteur (bloc C).

La loss dit si le modèle prédit bien les mots de la réponse, mais pas si le verdict est juste :
un modèle peut avoir une bonne loss et se tromper de verdict (PARFAIT au lieu de INCORRECT).
Ce fichier fait donc répondre le modèle à chaque conversation, puis compare le verdict généré
au verdict attendu. Il calcule la précision, le taux de format valide et la matrice de confusion
(un tableau qui montre, pour chaque verdict attendu, ce que le modèle a répondu).
"""

import argparse
import json
import random
import time
from pathlib import Path

from src.agents.baseline import parse_pedagogical_response
from src.training.dataset import load_split
from src.training.model import load_model_4bit, load_model_with_adapter
from src.training.train import generate_answer

# Les étiquettes possibles, dans l'ordre d'affichage de la matrice.
# HORS_PERIMETRE : le refus (le parseur renvoie alors verdict=None et valid_format=True).
# INVALIDE : une sortie qui ne respecte pas le contrat de sortie.
LABELS = ["PARFAIT", "PARTIEL", "INCORRECT", "HORS_PERIMETRE", "INVALIDE"]


def label_of(text: str) -> str:
    """Transforme une réponse du tuteur en une étiquette de verdict.

    Paramètres :
        text : réponse du tuteur (attendue ou générée par le modèle).

    Retour :
        "PARFAIT", "PARTIEL", "INCORRECT", "HORS_PERIMETRE" ou "INVALIDE".

    Exemple :
        label_of("[VERDICT: PARTIEL]\\nPresque.")  # "PARTIEL"
    """
    # C'est le même parseur que la baseline, donc le même contrat de sortie (contrat 2).
    parsed = parse_pedagogical_response(text)
    if not parsed["valid_format"]:
        return "INVALIDE"
    # format valide sans verdict = un refus [HORS_PERIMETRE]
    if parsed["verdict"] is None:
        return "HORS_PERIMETRE"
    return parsed["verdict"]


def select_indices(number_of_conversations: int, limit: int) -> list[int]:
    """Choisit au hasard les indices des conversations à évaluer (tous si limit vaut 0).

    Paramètres :
        number_of_conversations : nombre de conversations du split.
        limit : nombre d'indices à tirer ; 0 = tous.

    Retour :
        liste d'indices triés, la même à chaque appel (graine 0).

    Exemple :
        select_indices(118, 12)  # 12 indices entre 0 et 117
    """
    if limit <= 0:
        return list(range(number_of_conversations))
    # Val est trié par verdict (PARFAIT d'abord) : prendre les `limit` premiers ne testerait qu'un seul verdict,
    # et un modèle qui répond toujours PARFAIT aurait 100 %. Un tirage au hasard mélange les verdicts.
    # La graine fixe donne les mêmes exemples à chaque lancement, donc on peut comparer base et adaptateur.
    return sorted(random.Random(0).sample(range(number_of_conversations), limit))


def verdict_accuracy(model, tokenizer, conversations: list[list[dict]], indices: list[int] | None = None) -> dict:
    """Fait répondre le modèle à chaque conversation et compare son verdict au verdict attendu.

    Paramètres :
        model : modèle 4 bits, avec ou sans adaptateur LoRA.
        tokenizer : tokenizer du modèle.
        conversations : conversations de load_split (system, user, assistant).
        indices : position de chaque conversation dans le split complet (None : 0, 1, 2...) ;
            sert à retrouver l'exemple d'origine dans les lignes détaillées.

    Retour :
        {"number_of_examples", "number_of_correct", "accuracy", "valid_format_rate",
        "confusion", "rows"} ; confusion[attendu][prédit] = nombre d'exemples ;
        rows = une ligne {"index", "expected", "predicted", "correct"} par exemple
        (index = position dans le split complet).

    Exemple :
        verdict_accuracy(model, tokenizer, load_split("val")[:6])["accuracy"]  # un float entre 0 et 1
    """
    if indices is None:
        indices = list(range(len(conversations)))

    confusion = {}
    rows = []
    number_of_correct = 0
    number_of_valid = 0

    for position, conversation in enumerate(conversations):
        # conversation[:-1] = system + user, SANS la réponse du tuteur :
        # on ne donne jamais la réponse attendue au modèle, sinon il n'aurait qu'à la recopier.
        # La génération est gloutonne : la mesure est reproductible.
        generated_answer = generate_answer(model, tokenizer, conversation[:-1])

        expected = label_of(conversation[-1]["content"])
        predicted = label_of(generated_answer)
        is_correct = expected == predicted

        if is_correct:
            number_of_correct += 1
        # HORS_PERIMETRE compte comme valide (c'est un refus au bon format) ; seul INVALIDE ne l'est pas
        if predicted != "INVALIDE":
            number_of_valid += 1

        # ligne = verdict attendu, colonne = verdict prédit ; la diagonale = les réussites
        confusion.setdefault(expected, {})
        confusion[expected][predicted] = confusion[expected].get(predicted, 0) + 1
        index = indices[position]
        rows.append({"index": index, "expected": expected, "predicted": predicted, "correct": is_correct})
        print(f"exemple {position + 1}/{len(conversations)} (val n°{index}) : attendu {expected}, prédit {predicted}")

    number_of_examples = len(conversations)
    return {
        "number_of_examples": number_of_examples,
        "number_of_correct": number_of_correct,
        "accuracy": number_of_correct / number_of_examples,
        "valid_format_rate": number_of_valid / number_of_examples,
        "confusion": confusion,
        "rows": rows,
    }


def print_confusion(confusion: dict) -> None:
    """Affiche la matrice de confusion : une ligne par verdict attendu, une colonne par verdict prédit.

    Paramètres :
        confusion : confusion[attendu][prédit] = nombre d'exemples (sortie de verdict_accuracy).

    Retour :
        rien (affiche dans le terminal).

    Exemple :
        print_confusion({"PARFAIT": {"PARFAIT": 2, "PARTIEL": 1}})
    """
    print("matrice de confusion (lignes = attendu, colonnes = prédit) :")
    print(" " * 15 + "".join(f"{label:>15}" for label in LABELS))
    for expected in LABELS:
        # on saute les verdicts absents des données, leur ligne ne contiendrait que des 0
        if expected not in confusion:
            continue
        counts = "".join(f"{confusion[expected].get(predicted, 0):>15}" for predicted in LABELS)
        print(f"{expected:<15}" + counts)


def self_test() -> None:
    """Teste label_of sans charger de modèle (pas de GPU nécessaire).

    Retour :
        rien (lève AssertionError si un cas échoue).
    """
    assert label_of("[VERDICT: PARTIEL]\nx") == "PARTIEL"
    assert label_of("[HORS_PERIMETRE]\nx") == "HORS_PERIMETRE"
    assert label_of("bla") == "INVALIDE"
    print("self-test label_of : OK")


if __name__ == "__main__":
    # Lancer depuis la racine du dépôt (--limit 12 : 12 exemples tirés au hasard) :
    # python -m src.training.evaluate_verdicts --adapter adapters/v0.2/best --split val --limit 12 --output fichier.json
    parser = argparse.ArgumentParser(description="Mesure la précision du verdict du tuteur.")
    parser.add_argument("--adapter", default="none", help='dossier de l\'adaptateur LoRA, ou "none" pour le modèle de base')
    parser.add_argument("--split", default="val", help='"train", "val" ou "test"')
    parser.add_argument("--limit", type=int, default=0, help="nombre d'exemples tirés au hasard (graine 0) ; 0 = tous")
    parser.add_argument("--output", default=None, help="fichier JSON où écrire le résultat (optionnel)")
    parser.add_argument("--self-test", action="store_true", help="teste label_of puis s'arrête, sans charger de modèle")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        raise SystemExit(0)

    all_conversations = load_split(args.split)
    selected_indices = select_indices(len(all_conversations), args.limit)
    conversations = [all_conversations[index] for index in selected_indices]

    if args.adapter == "none":
        model, tokenizer = load_model_4bit()
    else:
        model, tokenizer = load_model_with_adapter(args.adapter)

    start_time = time.perf_counter()
    result = verdict_accuracy(model, tokenizer, conversations, selected_indices)
    total_seconds = time.perf_counter() - start_time

    number_of_examples = result["number_of_examples"]
    print(f"précision du verdict : {result['number_of_correct']}/{number_of_examples} ({100 * result['accuracy']:.1f} %)")
    print(f"taux de format valide : {100 * result['valid_format_rate']:.1f} %")
    print_confusion(result["confusion"])
    print(f"temps : {total_seconds:.1f} s ({total_seconds / number_of_examples:.1f} s par exemple)")

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"résultat écrit dans {output_path}")
