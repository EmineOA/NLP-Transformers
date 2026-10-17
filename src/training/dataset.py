"""Préparation des données pour l'entraînement QLoRA (bloc C).

Ce fichier lit les conversations du dataset (fichier JSONL + fichier CSV) et les
sépare en deux groupes : "train" (le modèle apprend dessus) et "val" (on vérifie
qu'il généralise). Il mesure aussi la longueur des conversations en tokens, puis
transforme chaque conversation en un exemple que le modèle peut apprendre :
la consigne est masquée, seule la réponse du tuteur compte pour l'apprentissage.
"""

import csv
import json
from pathlib import Path

# src/training/dataset.py -> on remonte de deux dossiers pour atteindre la racine du dépôt
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "dialogue" / "classification"
SPLITS = ("train", "val")


def load_split(split: str) -> list[list[dict]]:
    """Charge les conversations d'un split du dataset de classification.

    Paramètres :
        split : "train" ou "val" (colonne `split` de dataset_source.csv).

    Retour :
        liste de conversations ; chaque conversation est la liste `messages`
        (system, user, assistant) de dataset_messages.jsonl.

    Exemple :
        len(load_split("val"))  # 118
    """
    if split not in SPLITS:
        raise ValueError(f"split inconnu : {split} (attendu : {SPLITS})")

    # Étape 1 : lire le JSONL (une conversation par ligne)
    all_conversations = []
    with open(DATA_DIR / "dataset_messages.jsonl", encoding="utf-8") as file:
        for line in file:
            if line.strip() == "":
                continue  # on saute les lignes vides, elles ne contiennent pas de JSON
            line_data = json.loads(line)
            all_conversations.append(line_data["messages"])

    # Étape 2 : lire le CSV (il indique, pour chaque ligne, si elle est en train ou en val)
    # Excel ajoute un caractère invisible (BOM) au début du CSV ; sans utf-8-sig, la 1re colonne ne s'appellerait pas "id".
    # delimiter=";" : le CSV utilise le point-virgule (format Excel français), car les textes contiennent des virgules
    with open(DATA_DIR / "dataset_source.csv", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file, delimiter=";"))

    # la ligne i du JSONL correspond à la ligne i du CSV : un décalage mélangerait train et val
    number_of_conversations = len(all_conversations)
    number_of_rows = len(rows)
    if number_of_conversations != number_of_rows:
        raise ValueError(f"JSONL ({number_of_conversations}) et CSV ({number_of_rows}) n'ont pas le même nombre de lignes")

    # Étape 3 : ne garder que les conversations dont la ligne du CSV porte le bon split
    selected_conversations = []
    for index in range(number_of_conversations):
        if rows[index]["split"] == split:
            selected_conversations.append(all_conversations[index])
    return selected_conversations


def token_lengths(conversations: list[list[dict]], tokenizer) -> list[int]:
    """Compte les tokens de chaque conversation, telle que le modèle la lira.

    Un token est un morceau de texte (un mot ou une partie de mot) : c'est l'unité
    que le modèle lit, et sa mémoire est limitée à un nombre maximal de tokens.

    Paramètres :
        conversations : sortie de load_split.
        tokenizer : tokenizer du modèle (AutoTokenizer de transformers).

    Retour :
        une longueur (nombre de tokens) par conversation, dans le même ordre.

    Exemple :
        max(token_lengths(load_split("train"), tokenizer))  # longueur la plus grande
    """
    lengths = []
    for messages in conversations:
        # Le chat template est la mise en forme que le modèle attend (balises <|im_start|>…).
        # On mesure ce texte-là, et pas le JSON brut, car c'est lui que le modèle reçoit
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        # add_special_tokens=False : les balises sont déjà dans le texte, on ne les ajoute pas deux fois
        # ["input_ids"] = la liste des numéros des tokens (chaque token a un numéro dans le vocabulaire)
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        lengths.append(len(token_ids))
    return lengths


IGNORE_INDEX = -100  # valeur que PyTorch exclut du calcul de la loss
MAX_LENGTH = 4096  # conversation la plus longue mesurée : 3 638 tokens


def tokenize_example(messages: list[dict], tokenizer, max_length: int = MAX_LENGTH) -> dict:
    """Transforme une conversation en exemple d'entraînement : seule la réponse du tuteur compte.

    Paramètres :
        messages : une conversation (system, user, assistant) de load_split.
        tokenizer : tokenizer du modèle.
        max_length : nombre maximal de tokens ; au-delà, erreur (on ne coupe jamais).

    Retour :
        {"input_ids", "attention_mask", "labels"}, trois listes de même longueur ;
        labels vaut -100 sur le prompt et recopie les tokens de la réponse du tuteur.

    Exemple :
        tokenize_example(load_split("train")[0], tokenizer)["labels"][0]  # -100
    """
    # Le prompt = tout ce que le modèle reçoit avant de répondre (system + user).
    # add_generation_prompt=True ajoute à la fin "<|im_start|>assistant\n", l'invitation à répondre
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)

    # la réponse doit venir juste après le prompt, sinon on ne pourrait pas les séparer
    if not full_text.startswith(prompt_text):
        raise ValueError("le chat template ne place pas la réponse après le prompt")

    # on tokenise les deux morceaux séparément pour connaître la frontière exacte entre prompt et réponse
    # answer_text = la réponse du tuteur SUIVIE de la balise de fin <|im_end|> : le modèle apprend donc aussi à s'arrêter
    answer_text = full_text[len(prompt_text):]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + answer_ids

    number_of_tokens = len(input_ids)
    if number_of_tokens > max_length:
        # couper enlèverait la fin, c'est-à-dire la réponse du tuteur qu'on veut apprendre
        raise ValueError(f"{number_of_tokens} tokens > max_length {max_length}")

    # attention_mask : 1 = "ce token est un vrai token" (il n'y a pas de padding (remplissage) ici)
    attention_mask = [1] * number_of_tokens

    # labels : ce que le modèle doit deviner. -100 (IGNORE_INDEX) = "ne pas apprendre ce token",
    # donc le prompt est ignoré et seule la réponse du tuteur est apprise.
    # labels a la même longueur que input_ids ; le décalage d'un cran (prédire le token suivant)
    # est fait par le modèle lui-même, on ne décale rien ici
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


if __name__ == "__main__":
    # Lancer depuis la racine du dépôt : python -m src.training.dataset
    # Ces imports sont lents (et inutiles pour le reste du fichier) : on ne les fait qu'ici
    from transformers import AutoTokenizer

    from src.llm.model import DEFAULT_MODEL_NAME

    # Test 1 : chargement des deux splits
    train = load_split("train")
    val = load_split("val")
    # 885 train + 118 val = 1 003 conversations (split fixé par le bloc A)
    assert len(train) == 885 and len(val) == 118
    for messages in train + val:
        assert messages[-1]["role"] == "assistant"  # le dernier message est la réponse du tuteur
    print(f"load_split : {len(train)} train, {len(val)} val, chaque conversation finit par l'assistant")

    # Test 2 : longueurs en tokens
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    for name, conversations in (("train", train), ("val", val)):
        lengths = token_lengths(conversations, tokenizer)
        assert len(lengths) == len(conversations)
        assert min(lengths) > 0
        average_length = sum(lengths) // len(lengths)
        print(f"token_lengths {name} : min {min(lengths)}, moyenne {average_length}, max {max(lengths)}")

    # Test 3 : un exemple d'entraînement complet
    example = tokenize_example(train[0], tokenizer)
    assert len(example["input_ids"]) == len(example["attention_mask"]) == len(example["labels"])

    # on sépare les labels masqués (-100) des labels appris (vrais tokens de la réponse)
    learned_tokens = []
    for label in example["labels"]:
        if label != IGNORE_INDEX:
            learned_tokens.append(label)
    number_of_learned_tokens = len(learned_tokens)
    number_of_masked_tokens = len(example["labels"]) - number_of_learned_tokens

    # tout le prompt est masqué, d'un seul bloc, au début
    start_of_labels = example["labels"][:number_of_masked_tokens]
    assert start_of_labels == [IGNORE_INDEX] * number_of_masked_tokens
    print(f"tokenize_example : {number_of_masked_tokens} tokens masqués, {number_of_learned_tokens} appris")

    learned_text = tokenizer.decode(learned_tokens)
    print("réponse apprise :", repr(learned_text))
    # startswith et pas == : le texte décodé se termine par la balise de fin <|im_end|>
    assert learned_text.startswith(train[0][-1]["content"])
