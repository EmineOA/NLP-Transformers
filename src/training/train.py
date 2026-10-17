"""Boucle d'entraînement QLoRA (bloc C).

Ce fichier contient la boucle qui fait apprendre le modèle : pour chaque exemple,
le modèle essaie de prédire la réponse du tuteur, on mesure son erreur (la loss),
puis on corrige légèrement A et B, les petites matrices LoRA ajoutées par add_lora
(seules entraînées, le reste du modèle reste gelé).
Le test final vérifie que la boucle marche : le modèle doit apprendre par cœur un petit lot.
"""

import torch

from src.agents.baseline import parse_pedagogical_response
from src.training.dataset import load_split, tokenize_example
from src.training.model import LORA_TARGET_MODULES, add_lora, load_model_4bit, load_model_with_adapter

# Taille du pas de correction : trop grand, le modèle diverge ; trop petit, il n'apprend presque rien.
# 2e-4 : valeur utilisée dans l'article QLoRA pour les modèles 7B et 13B, et valeur courante pour LoRA ;
# à ajuster plus tard si besoin.
LEARNING_RATE = 2e-4


def train_steps(model, examples: list[dict], learning_rate: float = LEARNING_RATE, num_epochs: int = 1) -> list[float]:
    """Entraîne le modèle, un exemple à la fois, et retourne la loss de chaque pas.

    La loss est le nombre qui mesure l'erreur du modèle : plus il est petit, mieux le modèle prédit.
    Une epoch est un passage complet sur tous les exemples.

    Paramètres :
        model : modèle 4 bits avec adaptateurs LoRA (sortie de add_lora).
        examples : exemples déjà tokenisés (sortie de tokenize_example).
        learning_rate : taille du pas de correction.
        num_epochs : nombre de passages sur les exemples.

    Retour :
        liste de loss (float), une par pas, dans l'ordre.

    Exemple :
        len(train_steps(model, examples, num_epochs=2))  # 2 * len(examples)
    """
    # Le gradient traverse les poids gelés du modèle 4 bits pour atteindre A et B, mais on ne met à jour
    # que A et B : l'optimiseur AdamW (l'algorithme qui applique les corrections) ne reçoit donc
    # que les paramètres entraînables (requires_grad).
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    # mode entraînement : active le dropout des adaptateurs LoRA, défini dans add_lora
    model.train()

    losses = []
    for epoch in range(1, num_epochs + 1):
        for example in examples:
            # [1, longueur] : 1 = taille du batch, un exemple à la fois (donc pas de padding)
            input_ids = torch.tensor([example["input_ids"]], device=model.device)
            attention_mask = torch.tensor([example["attention_mask"]], device=model.device)
            labels = torch.tensor([example["labels"]], device=model.device)

            # Un pas d'entraînement, en 4 temps :
            # 1. le modèle prédit chaque token et calcule l'erreur sur la seule réponse du tuteur :
            #    les -100 de labels sont les tokens ignorés dans la loss (consigne et messages de l'élève)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # quand on donne labels, le modèle calcule lui-même la loss : l'erreur moyenne sur les tokens à prédire
            loss = outputs.loss
            # 2. backward calcule, pour chaque valeur de A et B, comment la loss change si on la modifie un peu
            #    (c'est le gradient : il pointe vers là où la loss AUGMENTE).
            #    backward n'a pas d'argument car il remonte tous les calculs qui ont produit loss
            #    (PyTorch les a enregistrés).
            loss.backward()
            # 3. step déplace A et B dans le sens OPPOSÉ au gradient, d'un petit pas, pour faire baisser la loss
            optimizer.step()
            # 4. zero_grad remet les gradients à zéro, sinon ils s'additionneraient d'un exemple à l'autre
            optimizer.zero_grad()

            # .item() récupère le nombre seul, sans garder tout le graphe de calcul en mémoire
            losses.append(loss.item())
            print(f"epoch {epoch}, pas {len(losses)} : loss {losses[-1]:.4f}")
    return losses


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


if __name__ == "__main__":
    # Lancer depuis la racine du dépôt : python -m src.training.train
    import time
    from pathlib import Path

    NUM_EXAMPLES = 5
    NUM_EPOCHS = 4

    # Test "apprendre par cœur un petit lot" : si le pipeline marche, le modèle doit mémoriser 5 réponses
    # et la loss doit baisser entre le 1er et le dernier passage ; sinon il y a un bug.
    model, tokenizer = load_model_4bit()
    model = add_lora(model)

    conversations = load_split("train")[:NUM_EXAMPLES]
    examples = [tokenize_example(messages, tokenizer) for messages in conversations]

    # but de la mesure de mémoire : vérifier que l'entraînement tient dans les 12 Go de la RTX 4070
    # et les 16 Go de la T4 de Colab
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    losses = train_steps(model, examples, num_epochs=NUM_EPOCHS)
    total_seconds = time.perf_counter() - start_time

    assert len(losses) == NUM_EXAMPLES * NUM_EPOCHS  # une loss par pas
    first_pass_average = sum(losses[:NUM_EXAMPLES]) / NUM_EXAMPLES
    last_pass_average = sum(losses[-NUM_EXAMPLES:]) / NUM_EXAMPLES
    peak_memory_in_gb = torch.cuda.max_memory_allocated() / 1024**3  # octets -> Go

    print(f"loss moyenne, 1er passage : {first_pass_average:.4f}")
    print(f"loss moyenne, {NUM_EPOCHS}e passage : {last_pass_average:.4f}")
    print(f"temps total : {total_seconds:.1f} s")
    print(f"mémoire GPU maximale : {peak_memory_in_gb:.2f} Go")

    # Ce test vérifie seulement que la boucle marche (gradients qui circulent, A et B qui bougent,
    # labels cohérents). Le modèle revoit 4 fois les mêmes 5 exemples : la baisse vient de la mémorisation.
    # Elle ne prouve PAS que le modèle sera bon sur des exemples nouveaux ; cela se mesure plus tard
    # avec la loss de validation (118 exemples val).
    assert last_pass_average < first_pass_average

    # Dossier ignoré par git (voir .gitignore). Nom "test" : cet adaptateur n'a vu que 5 exemples.
    ADAPTER_DIR = "adapters/test_5_exemples"

    # save_pretrained sur un modèle LoRA sauvegarde seulement A et B, pas les 3 milliards de poids de base
    model.save_pretrained(ADAPTER_DIR)

    total_bytes = 0
    for file_path in Path(ADAPTER_DIR).iterdir():
        total_bytes += file_path.stat().st_size
    print(f"taille de l'adaptateur : {total_bytes / 1024**2:.1f} Mo")  # octets -> Mo

    # On libère le GPU : sans cela, le modèle entraîné et le modèle rechargé seraient en mémoire en même temps.
    # del supprime notre variable ; empty_cache rend au GPU la mémoire que PyTorch gardait en réserve.
    # Recharger un modèle neuf prouve que les fichiers sur le disque suffisent.
    del model
    torch.cuda.empty_cache()

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
