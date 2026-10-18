"""Boucle d'entraînement QLoRA (bloc C).

Ce fichier contient la boucle qui fait apprendre le modèle : pour chaque exemple,
le modèle essaie de prédire la réponse du tuteur, on mesure son erreur (la loss),
puis on corrige légèrement A et B, les petites matrices LoRA ajoutées par add_lora
(seules entraînées, le reste du modèle reste gelé).
Le test final vérifie que la boucle marche : le modèle doit apprendre par cœur un petit lot.
"""

import random
import time
from pathlib import Path

import torch

from src.agents.baseline import parse_pedagogical_response
from src.training.dataset import load_split, tokenize_example
from src.training.model import LORA_TARGET_MODULES, add_lora, load_model_4bit, load_model_with_adapter

# Taille du pas de correction : trop grand, le modèle diverge ; trop petit, il n'apprend presque rien.
# 2e-4 : valeur utilisée dans l'article QLoRA pour les modèles 7B et 13B, et valeur courante pour LoRA ;
# à ajuster plus tard si besoin.
LEARNING_RATE = 2e-4

# On additionne les gradients de 8 exemples avant de corriger A et B : la direction est plus fiable
# qu'avec un seul exemple, et la mémoire reste celle d'un exemple à la fois.
# Cela équivaut à un batch de 8 pour la moyenne des gradients
# (au token près : les exemples n'ont pas tous la même longueur).
ACCUMULATION_STEPS = 8

# Graine du hasard : même ordre de mélange à chaque lancement, donc résultats reproductibles.
SEED = 42


def train_steps(
    model,
    examples: list[dict],
    learning_rate: float = LEARNING_RATE,
    num_epochs: int = 1,
    accumulation_steps: int = ACCUMULATION_STEPS,
    seed: int = SEED,
    optimizer=None,
    epoch_label: int = None,
    on_optimizer_step=None,
    start_step: int = 0,
) -> tuple[list[float], int]:
    """Entraîne le modèle, par groupes d'exemples, et retourne les loss et le nombre de corrections.

    La loss est le nombre qui mesure l'erreur du modèle : plus il est petit, mieux le modèle prédit.
    Une epoch est un passage complet sur tous les exemples, dans un ordre mélangé.
    Un groupe est une série d'exemples dont on additionne les gradients avant une seule correction de A et B.

    Paramètres :
        model : modèle 4 bits avec adaptateurs LoRA (sortie de add_lora).
        examples : exemples déjà tokenisés (sortie de tokenize_example) ; la liste n'est pas modifiée.
        learning_rate : taille du pas de correction.
        num_epochs : nombre de passages sur les exemples.
        accumulation_steps : nombre d'exemples par groupe (le dernier groupe peut être plus petit).
        seed : graine du mélange, pour retrouver le même ordre à chaque lancement.
        optimizer : optimiseur AdamW déjà créé (None : train_steps en crée un neuf avec learning_rate).
        epoch_label : numéro d'epoch affiché dans les messages (None : on affiche 1, 2, 3... selon la boucle).
            Utile quand l'appelant lance une epoch à la fois : sans lui, chaque appel afficherait "epoch 1".
        on_optimizer_step : fonction appelée après chaque correction avec (step_number, group_loss),
            où group_loss est la moyenne des loss (non divisées) des exemples du groupe.
            Si elle renvoie True, l'entraînement s'arrête tout de suite (None : pas de fonction).
        start_step : numéro de la dernière correction déjà faite avant cet appel (0 au départ) ;
            l'appelant qui lance une epoch à la fois le passe pour que step_number continue d'une epoch à l'autre.

    Retour :
        (losses, number_of_optimizer_steps) : la loss de chaque exemple (float, dans l'ordre vu)
        et le nombre de corrections de A et B faites dans cet appel (le dernier step_number est
        start_step + number_of_optimizer_steps).

    Exemple :
        losses, steps = train_steps(model, examples, num_epochs=2, accumulation_steps=1)
        # len(losses) == steps == 2 * len(examples)
    """
    # Le gradient traverse les poids gelés du modèle 4 bits pour atteindre A et B, mais on ne met à jour
    # que A et B : l'optimiseur AdamW (l'algorithme qui applique les corrections) ne reçoit donc
    # que les paramètres entraînables (requires_grad).
    # Pour l'entraînement complet, l'appelant passe le même optimiseur d'une epoch à l'autre :
    # AdamW garde en mémoire une moyenne des gradients passés, et le recréer à chaque epoch effacerait cette mémoire.
    if optimizer is None:
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    # mode entraînement : active le dropout des adaptateurs LoRA, défini dans add_lora
    model.train()

    losses = []
    number_of_optimizer_steps = 0
    for epoch in range(1, num_epochs + 1):
        # On mélange une liste d'indices et non la liste reçue, pour ne pas la modifier chez l'appelant.
        # Sans mélange, le modèle verrait toutes les questions d'un même concept d'affilée.
        # seed + epoch : un ordre différent à chaque epoch, mais le même à chaque lancement.
        # random.Random(...) crée un générateur de hasard local : le hasard global de Python n'est pas touché.
        indices = list(range(len(examples)))
        random.Random(seed + epoch).shuffle(indices)

        # On découpe l'ordre mélangé en groupes de accumulation_steps exemples ;
        # le dernier groupe peut être plus court (par exemple 10 exemples en groupes de 4 : 4 + 4 + 2).
        for group_start in range(0, len(indices), accumulation_steps):
            group_indices = indices[group_start:group_start + accumulation_steps]
            # vraie taille du groupe : le dernier groupe peut avoir moins de accumulation_steps exemples
            group_size = len(group_indices)

            for index in group_indices:
                example = examples[index]
                # [1, longueur] : 1 = taille du batch, un exemple à la fois (donc pas de padding)
                input_ids = torch.tensor([example["input_ids"]], device=model.device)
                attention_mask = torch.tensor([example["attention_mask"]], device=model.device)
                labels = torch.tensor([example["labels"]], device=model.device)

                # Un groupe d'entraînement, en 5 temps (les temps 1 à 3 se répètent pour chaque exemple) :
                # 1. le modèle prédit chaque token. Avec labels, il renvoie lui-même outputs.loss :
                #    la moyenne de l'erreur sur les tokens de la réponse du tuteur.
                #    Les -100 de labels sont les tokens ignorés (consigne et messages de l'élève).
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                # 2. backward ADDITIONNE les gradients d'un exemple à l'autre. Diviser la loss par la taille
                #    du groupe donne donc leur MOYENNE. On divise par la vraie taille pour que le dernier groupe,
                #    plus petit, compte juste.
                loss = outputs.loss / group_size
                # 3. backward calcule, pour chaque valeur de A et B, comment la loss change si on la modifie un peu
                #    (c'est le gradient : il pointe vers là où la loss AUGMENTE).
                #    backward n'a pas d'argument car il remonte tous les calculs qui ont produit loss
                #    (PyTorch les a enregistrés).
                loss.backward()

                # On enregistre la loss NON divisée, sinon les valeurs affichées dépendraient de la taille du groupe.
                # .item() récupère le nombre seul, sans garder tout le graphe de calcul en mémoire
                losses.append(outputs.loss.item())
                shown_epoch = epoch if epoch_label is None else epoch_label
                print(f"epoch {shown_epoch}, exemple {len(losses)} : loss {losses[-1]:.4f}")

            # Temps 4 et 5 : une fois par groupe (hors de la boucle sur les exemples du groupe).
            # 4. step déplace A et B dans le sens OPPOSÉ au gradient (ici la moyenne du groupe),
            #    d'un petit pas, pour faire baisser la loss
            optimizer.step()
            # 5. zero_grad remet les gradients à zéro, sinon ils s'additionneraient d'un groupe à l'autre
            optimizer.zero_grad()
            number_of_optimizer_steps += 1

            # Les dernières group_size loss sont celles du groupe qui vient d'être corrigé.
            if on_optimizer_step is not None:
                group_loss = sum(losses[-group_size:]) / group_size
                stop_requested = on_optimizer_step(start_step + number_of_optimizer_steps, group_loss)
                # Sortir ici (et pas seulement de la boucle des groupes) arrête aussi les epochs suivantes.
                if stop_requested:
                    return losses, number_of_optimizer_steps
    return losses, number_of_optimizer_steps


def evaluate_loss(model, examples: list[dict]) -> float:
    """Mesure la loss moyenne du modèle sur des exemples, sans l'entraîner.

    Avec les exemples val (jamais vus à l'entraînement), c'est la "loss de validation" :
    le seul chiffre qui dit si le modèle progresse sur des cas nouveaux.
    Si elle remonte alors que la loss d'entraînement baisse, c'est du surapprentissage :
    le modèle apprend ses exemples par cœur au lieu d'apprendre à bien répondre.

    Paramètres :
        model : modèle 4 bits avec adaptateurs LoRA (sortie de add_lora).
        examples : exemples déjà tokenisés (sortie de tokenize_example).

    Retour :
        loss moyenne (float) sur tous les exemples.

    Exemple :
        evaluate_loss(model, val_examples)  # un float : la loss moyenne
    """
    # eval() : coupe le dropout, pour une mesure stable
    model.eval()

    total_loss = 0.0
    # no_grad : PyTorch n'enregistre pas les calculs pour les gradients (moins de mémoire).
    # A et B ne bougent pas car on n'appelle jamais optimizer.step().
    with torch.no_grad():
        for example in examples:
            # [1, longueur] : 1 = taille du batch, un exemple à la fois (comme dans train_steps)
            input_ids = torch.tensor([example["input_ids"]], device=model.device)
            attention_mask = torch.tensor([example["attention_mask"]], device=model.device)
            labels = torch.tensor([example["labels"]], device=model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

    # on remet le mode entraînement : la boucle complète appellera evaluate_loss entre deux epochs
    model.train()
    return total_loss / len(examples)


def generate_answer(model, tokenizer, messages: list[dict], max_new_tokens: int = 200) -> str:
    """Fait répondre le modèle à une conversation et retourne seulement le texte généré.

    Paramètres :
        model : modèle 4 bits, avec ou sans adaptateur LoRA.
        tokenizer : tokenizer du modèle.
        messages : conversation SANS la réponse du tuteur (system, user).
        max_new_tokens : nombre maximal de tokens que le modèle a le droit d'écrire.

    Retour :
        la réponse du tuteur (str), sans le prompt et sans les balises spéciales.

    Exemple :
        generate_answer(model, tokenizer, load_split("val")[0][:-1])  # "[VERDICT: PARTIEL]\\n..."
    """
    # On s'arrête juste après "<|im_start|>assistant\n" : c'est là que le tuteur doit écrire
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # add_special_tokens=False : le chat template a déjà ajouté les balises spéciales
    inputs = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").to(model.device)

    # pas d'entraînement ici : sans gradients, on économise de la mémoire
    with torch.no_grad():
        # do_sample=False : mode glouton, à chaque pas le modèle prend le token le plus probable
        # (l'argmax vu dans les notebooks du bloc B) ; le résultat est donc reproductible
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # generate renvoie le prompt SUIVI de la réponse : on saute le prompt.
    # output_ids[0] : un seul prompt dans le batch ; shape[1] : nombre de tokens du prompt
    prompt_length = inputs["input_ids"].shape[1]
    new_ids = output_ids[0][prompt_length:]
    # skip_special_tokens=True retire les balises comme <|im_end|> du texte
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# Dossier ignoré par git (voir .gitignore). Nom "test" : cet adaptateur n'a vu que 5 exemples.
ADAPTER_DIR = "adapters/test_5_exemples"


def test_memorize_small_batch():
    """Test : le modèle apprend par cœur 5 exemples, et l'adaptateur est sauvegardé dans ADAPTER_DIR."""
    NUM_EXAMPLES = 5
    NUM_EPOCHS = 4
    # un sous-ensemble de val pour que le test reste rapide ; l'entraînement complet utilisera les 118
    NUM_VAL_EXAMPLES = 20

    # Si le pipeline marche, le modèle doit mémoriser 5 réponses et la loss doit baisser
    # entre le 1er et le dernier passage ; sinon il y a un bug.
    model, tokenizer = load_model_4bit()
    model = add_lora(model)

    conversations = load_split("train")[:NUM_EXAMPLES]
    examples = [tokenize_example(messages, tokenizer) for messages in conversations]

    val_conversations = load_split("val")[:NUM_VAL_EXAMPLES]
    val_examples = [tokenize_example(messages, tokenizer) for messages in val_conversations]

    # but de la mesure de mémoire : vérifier que l'entraînement tient dans les 12 Go de la RTX 4070
    # et les 16 Go de la T4 de Colab
    torch.cuda.reset_peak_memory_stats()

    # La copie se fait AVANT evaluate_loss, sinon on ne pourrait pas comparer avant/après.
    # On copie TOUS les paramètres LoRA (environ 115 Mo) : une seule matrice prouverait trop peu.
    # detach : la copie n'est plus reliée aux calculs de PyTorch ;
    # clone : vraie copie des valeurs, pas un simple lien.
    lora_copies = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_copies[name] = param.detach().clone()

    val_loss_before = evaluate_loss(model, val_examples)

    # prouve que l'évaluation n'entraîne pas : chaque paramètre LoRA doit être identique à sa copie.
    # torch.equal = toutes les valeurs identiques (== donnerait un tableau de vrai/faux)
    for name, param in model.named_parameters():
        if "lora_" in name:
            assert torch.equal(lora_copies[name], param), f"{name} a changé pendant evaluate_loss"
    print(f"paramètres LoRA inchangés par evaluate_loss : OK ({len(lora_copies)} tenseurs)")

    # le chrono ne couvre que l'entraînement, pour rester comparable aux mesures précédentes
    start_time = time.perf_counter()
    # accumulation_steps=1 : une correction par exemple, comme avant l'accumulation
    losses, number_of_optimizer_steps = train_steps(model, examples, num_epochs=NUM_EPOCHS, accumulation_steps=1)
    total_seconds = time.perf_counter() - start_time

    val_loss_after = evaluate_loss(model, val_examples)

    assert len(losses) == NUM_EXAMPLES * NUM_EPOCHS  # une loss par exemple vu
    assert number_of_optimizer_steps == NUM_EXAMPLES * NUM_EPOCHS  # 5 exemples * 4 epochs = 20 corrections
    first_pass_average = sum(losses[:NUM_EXAMPLES]) / NUM_EXAMPLES
    last_pass_average = sum(losses[-NUM_EXAMPLES:]) / NUM_EXAMPLES
    peak_memory_in_gb = torch.cuda.max_memory_allocated() / 1024**3  # octets -> Go

    # Pas d'assert sur le sens de la loss val : baisse, stagnation ou hausse sont des résultats possibles, pas des bugs.
    # Val avant = modèle de base (au départ B vaut 0, donc LoRA ne change rien).
    # Val qui monte pendant que train baisse = surapprentissage ;
    # sur 20 exemples, de petites variations peuvent aussi être du bruit.
    assert val_loss_before > 0 and val_loss_after > 0  # la mesure renvoie bien une loss

    print(f"loss moyenne, 1er passage : {first_pass_average:.4f}")
    print(f"loss moyenne, {NUM_EPOCHS}e passage : {last_pass_average:.4f}")
    print(f"loss val avant : {val_loss_before:.4f}")
    print(f"loss val après : {val_loss_after:.4f}")
    print(f"temps total : {total_seconds:.1f} s")
    print(f"mémoire GPU maximale : {peak_memory_in_gb:.2f} Go")

    # Ce test vérifie seulement que la boucle marche (gradients qui circulent, A et B qui bougent,
    # labels cohérents). Le modèle revoit 4 fois les mêmes 5 exemples : la baisse vient de la mémorisation.
    # Elle ne prouve PAS que le modèle sera bon sur des exemples nouveaux ; cela se mesure plus tard
    # avec la loss de validation (118 exemples val).
    assert last_pass_average < first_pass_average

    # save_pretrained sur un modèle LoRA sauvegarde seulement A et B, pas les 3 milliards de poids de base
    model.save_pretrained(ADAPTER_DIR)

    total_bytes = 0
    for file_path in Path(ADAPTER_DIR).iterdir():
        total_bytes += file_path.stat().st_size
    print(f"taille de l'adaptateur : {total_bytes / 1024**2:.1f} Mo")  # octets -> Mo

    # On libère le GPU pour que le test suivant parte d'une mémoire vide.
    # del supprime notre variable ; empty_cache rend au GPU la mémoire que PyTorch gardait en réserve.
    del model
    torch.cuda.empty_cache()


def test_save_and_reload():
    """Test : l'adaptateur sauvegardé par test_memorize_small_batch se recharge et répond au bon format."""
    # Recharger un modèle neuf prouve que les fichiers sur le disque suffisent.
    model, tokenizer = load_model_with_adapter(ADAPTER_DIR)

    number_of_lora_parameters = 0
    for name, _ in model.named_parameters():
        if "lora_" in name:
            number_of_lora_parameters += 1

    # 2 matrices (A et B) par couche adaptée, 7 couches ciblées par bloc, 36 blocs dans Qwen2.5-3B
    number_of_adapted_layers = len(LORA_TARGET_MODULES) * model.config.num_hidden_layers
    expected = 2 * number_of_adapted_layers
    assert number_of_lora_parameters == expected
    print(f"adaptateur rechargé : {number_of_lora_parameters} paramètres LoRA (A et B de {number_of_adapted_layers} couches)")

    # Le premier exemple de val : le modèle ne l'a jamais vu à l'entraînement.
    # val_example = [system, user, tuteur] : [:-1] = la conversation sans la réponse du tuteur,
    # [-1] = la réponse attendue
    val_example = load_split("val")[0]
    generated_answer = generate_answer(model, tokenizer, val_example[:-1])
    expected_answer = val_example[-1]["content"]
    print("réponse générée :")
    print(generated_answer)
    print("réponse attendue :")
    print(expected_answer)

    # On vérifie le FORMAT (contrat 2), pas la qualité : cet adaptateur n'a vu que 5 exemples.
    # C'est le même parseur que la baseline, donc le même contrat de sortie.
    # Un refus [HORS_PERIMETRE] est valide : le parseur renvoie verdict=None et valid_format=True.
    # On teste donc valid_format et non verdict.
    # Si l'assert échoue, ce n'est pas forcément un bug de code : un adaptateur entraîné sur 5 exemples
    # peut mal respecter le format.
    parsed = parse_pedagogical_response(generated_answer)
    assert parsed["valid_format"], "la réponse ne respecte pas le contrat de sortie"
    print(f"format valide (verdict : {parsed['verdict']})")

    del model
    torch.cuda.empty_cache()


def test_accumulation():
    """Test : 10 exemples en groupes de 4 donnent 3 corrections (4 + 4 + 2) et 10 loss."""
    # Modèle neuf avec add_lora, pour deux raisons :
    # - le modèle rechargé est gelé et sans gradient checkpointing (pas assez de mémoire sur la RTX 4070) ;
    # - un modèle déjà entraîné rendrait ce test dépendant du précédent.
    model, tokenizer = load_model_4bit()
    model = add_lora(model)

    # Les exemples 5 à 14 de train : 10 exemples, différents des 5 du premier test
    conversations = load_split("train")[5:15]
    examples = [tokenize_example(messages, tokenizer) for messages in conversations]

    # 10 exemples en groupes de 4 : 4 + 4 + 2 = 3 groupes (le dernier est plus petit), donc 3 corrections
    losses, number_of_optimizer_steps = train_steps(model, examples, num_epochs=1, accumulation_steps=4)
    assert number_of_optimizer_steps == 3
    assert len(losses) == 10
    print("accumulation : 10 exemples, groupes de 4 → 3 corrections")

    del model
    torch.cuda.empty_cache()


def test_callback_and_stop():
    """Test : la fonction de rappel reçoit les bons numéros de pas, et renvoyer True arrête l'entraînement."""
    model, tokenizer = load_model_4bit()
    model = add_lora(model)

    # 6 exemples en groupes de 2 : 3 corrections par epoch
    conversations = load_split("train")[15:21]
    examples = [tokenize_example(messages, tokenizer) for messages in conversations]

    calls = []

    def record_and_stop_at_5(step_number, group_loss):
        calls.append((step_number, group_loss))
        return step_number == 5

    # start_step=2 : les pas sont numérotés 3, 4, 5... ; l'arrêt à 5 tombe au milieu de la 2e epoch
    losses, number_of_optimizer_steps = train_steps(
        model, examples, num_epochs=2, accumulation_steps=2, on_optimizer_step=record_and_stop_at_5, start_step=2,
    )
    assert [step for step, _ in calls] == [3, 4, 5]
    assert number_of_optimizer_steps == 3
    assert len(losses) == 6
    # la loss du groupe 1 est la moyenne de ses 2 exemples
    assert abs(calls[0][1] - sum(losses[:2]) / 2) < 1e-9
    print("rappel : pas 3, 4, 5 reçus, arrêt à 5 sur demande")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Lancer depuis la racine du dépôt : python -m src.training.train
    # L'ordre compte : test_save_and_reload lit l'adaptateur écrit par test_memorize_small_batch.
    test_memorize_small_batch()
    test_save_and_reload()
    test_accumulation()
    test_callback_and_stop()
