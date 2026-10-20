"""Préparation des données pour l'entraînement QLoRA (bloc C).

Ce fichier lit les conversations du dataset (fichier JSONL + fichier CSV) et les
sépare en trois groupes : "train" (le modèle apprend dessus), "val" (on vérifie
qu'il généralise et on choisit le meilleur adaptateur) et "test" (mesure finale,
jamais vue pendant l'entraînement ni le choix de l'adaptateur). Il mesure aussi la longueur des conversations en tokens, puis
transforme chaque conversation en un exemple que le modèle peut apprendre :
la consigne est masquée, seule la réponse du tuteur compte pour l'apprentissage.
"""

import csv
import json
from pathlib import Path

import torch

# src/training/dataset.py -> on remonte de deux dossiers pour atteindre la racine du dépôt
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "dialogue" / "classification"
SPLITS = ("train", "val", "test")


def load_split(split: str) -> list[list[dict]]:
    """Charge les conversations d'un split du dataset de classification.

    Paramètres :
        split : "train", "val" ou "test" (colonne `split` de dataset_source.csv).

    Retour :
        liste de conversations ; chaque conversation est la liste `messages`
        (system, user, assistant) de dataset_messages.jsonl.

    Exemple :
        len(load_split("val"))  # 177
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
    # est fait plus tard, dans per_example_loss (train.py), on ne décale rien ici
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def make_batch(examples: list[dict], pad_token_id: int) -> dict:
    """Regroupe plusieurs exemples en un lot : des tenseurs de même longueur, que le modèle lit en une fois.

    Les exemples n'ont pas la même longueur, or un tenseur est un tableau rectangulaire.
    On complète donc les plus courts à droite jusqu'à la longueur du plus long (le padding, du remplissage).

    Paramètres :
        examples : exemples de tokenize_example (au moins un).
        pad_token_id : numéro du token de remplissage (tokenizer.pad_token_id).

    Retour :
        {"input_ids", "attention_mask", "labels"}, trois tenseurs torch de forme [nombre d'exemples, longueur du plus long].

    Exemple :
        make_batch([court, long], 0)["attention_mask"][0]  # tensor([1, 1, 1, 0, 0, 0])
    """
    max_length = max(len(example["input_ids"]) for example in examples)

    input_ids, attention_mask, labels = [], [], []
    for example in examples:
        number_of_pads = max_length - len(example["input_ids"])
        # Le padding est du remplissage : il ne doit ni être lu, ni compter dans la loss.
        # input_ids : on met le token de remplissage, il faut bien une valeur valide pour le modèle.
        input_ids.append(example["input_ids"] + [pad_token_id] * number_of_pads)
        # attention_mask à 0 : le modèle ne regarde pas ces positions (le vrai texte ne "lit" jamais le padding).
        attention_mask.append(example["attention_mask"] + [0] * number_of_pads)
        # labels à -100 : ces positions sont ignorées dans le calcul de la loss.
        labels.append(example["labels"] + [IGNORE_INDEX] * number_of_pads)

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }


if __name__ == "__main__":
    # Lancer depuis la racine du dépôt : python -m src.training.dataset
    # Test 0 (sans GPU, sans tokenizer) : make_batch sur deux petits exemples fabriqués, de longueurs 3 et 5
    short = {"input_ids": [10, 11, 12], "attention_mask": [1, 1, 1], "labels": [-100, 11, 12]}
    long = {"input_ids": [20, 21, 22, 23, 24], "attention_mask": [1] * 5, "labels": [-100, -100, 22, 23, 24]}
    batch = make_batch([short, long], pad_token_id=99)
    assert batch["input_ids"].shape == (2, 5)
    assert batch["input_ids"][0].tolist() == [10, 11, 12, 99, 99]
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
    assert batch["labels"][0].tolist() == [-100, 11, 12, -100, -100]
    assert batch["input_ids"][1].tolist() == long["input_ids"]  # le plus long n'est pas modifié
    print("make_batch : padding à droite, attention_mask 0 et labels -100 sur le remplissage")

    # Ces imports sont lents (et inutiles pour le reste du fichier) : on ne les fait qu'ici
    from transformers import AutoTokenizer

    from src.llm.model import DEFAULT_MODEL_NAME

    # Test 1 : chargement des trois splits
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    # 649 train + 177 val + 177 test = 1 003 conversations (split fixé par le bloc A, par question entière)
    assert len(train) == 649 and len(val) == 177 and len(test) == 177
    for messages in train + val + test:
        assert messages[-1]["role"] == "assistant"  # le dernier message est la réponse du tuteur

    # une question présente dans deux splits ferait fuiter ses réponses de train vers val ou test
    with open(DATA_DIR / "dataset_source.csv", encoding="utf-8-sig", newline="") as file:
        source_rows = list(csv.DictReader(file, delimiter=";"))
    questions = {name: {row["question_id"] for row in source_rows if row["split"] == name} for name in SPLITS}
    assert not questions["train"] & questions["val"]
    assert not questions["train"] & questions["test"]
    assert not questions["val"] & questions["test"]
    print(f"load_split : {len(train)} train, {len(val)} val, {len(test)} test, aucune question partagée, "
          "chaque conversation finit par l'assistant")

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
