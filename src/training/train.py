"""Boucle d'entraînement QLoRA (bloc C).

Ce fichier contient la boucle qui fait apprendre le modèle : pour chaque exemple,
le modèle essaie de prédire la réponse du tuteur, on mesure son erreur (la loss),
puis on corrige légèrement A et B, les petites matrices LoRA ajoutées par add_lora
(seules entraînées, le reste du modèle reste gelé).
Le test final vérifie que la boucle marche : le modèle doit apprendre par cœur un petit lot.
"""

import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn.functional import cross_entropy

from src.agents.baseline import parse_pedagogical_response
from src.training.dataset import IGNORE_INDEX, load_split, make_batch, tokenize_example
from src.training.model import LORA_TARGET_MODULES, add_lora, load_model_4bit, load_model_with_adapter

# Taille du pas de correction : trop grand, le modèle diverge ; trop petit, il n'apprend presque rien.
# 2e-4 : valeur utilisée dans l'article QLoRA pour les modèles 7B et 13B, et valeur courante pour LoRA ;
# à ajuster plus tard si besoin.
LEARNING_RATE = 2e-4

# On additionne les gradients de 8 lots avant de corriger A et B : la direction est plus fiable
# qu'avec un seul exemple, et la mémoire reste celle d'un lot à la fois.
# Avec batch_size = 1 (défaut), cela équivaut à un batch de 8 exemples.
# Chaque exemple compte autant dans la moyenne, quelle que soit la longueur de sa réponse (voir per_example_loss).
ACCUMULATION_STEPS = 8

# Graine du hasard : même ordre de mélange à chaque lancement, donc résultats reproductibles.
SEED = 42


def per_example_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Calcule la loss de chaque exemple d'un lot, moyennée sur les tokens que cet exemple doit apprendre.

    Chaque exemple compte autant, quelle que soit la longueur de sa réponse. La loss moyenne du modèle
    (outputs.loss) est une moyenne sur TOUS les tokens du lot : une longue réponse y pèserait plus qu'une courte.

    Paramètres :
        logits : scores du modèle pour chaque token possible, forme [lot, longueur, vocabulaire].
        labels : tokens à deviner, forme [lot, longueur] ; -100 = token ignoré (prompt ou padding).

    Retour :
        un tenseur [lot] : la loss de chaque exemple (0 si l'exemple n'a aucun token à apprendre).

    Exemple :
        per_example_loss(torch.zeros(1, 3, 4), torch.tensor([[-100, 1, 2]]))  # tensor([1.3863]) = ln(4)
    """
    # Décalage d'un cran : le logit à la position i sert à prédire le token i+1.
    # On retire donc le dernier logit (il ne prédit rien) et le premier label (personne ne le prédit).
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    # On ne garde que les positions apprises (labels différents de -100) : la réponse du tuteur ne représente
    # que quelques % des tokens (le reste est le prompt). Le passage en float32 ne coûte donc que
    # (tokens appris) x 151 936 (taille du vocabulaire de Qwen) x 4 octets, et non lot x longueur x 151 936 x 4.
    # Le résultat est identique : les positions ignorées ne comptaient pas dans la loss.
    # float32 pour la cross_entropy : stabilité du log et de la somme.
    mask = shift_labels != IGNORE_INDEX
    selected_logits = shift_logits[mask].float()  # [nombre de tokens appris dans le lot, vocabulaire]
    selected_labels = shift_labels[mask]

    # reduction="none" : une loss par token, pour pouvoir faire nous-mêmes la moyenne par exemple.
    token_losses = cross_entropy(selected_logits, selected_labels, reduction="none")

    # Il faut retrouver à quel exemple appartient chaque token : row_index donne le numéro d'exemple (la ligne)
    # de chaque position retenue, dans le même ordre que token_losses.
    number_of_examples = labels.shape[0]
    row_index = torch.arange(number_of_examples, device=labels.device).unsqueeze(1).expand_as(mask)[mask]
    # index_add additionne les loss des tokens de chaque exemple : somme par exemple
    loss_sums = torch.zeros(number_of_examples, device=labels.device).index_add(0, row_index, token_losses)

    number_of_learned_tokens = mask.sum(dim=1)
    # clamp(min=1) : évite une division par 0 pour un exemple sans aucun token appris (sa loss vaut alors 0)
    return loss_sums / number_of_learned_tokens.clamp(min=1)


def forward_losses(model, examples: list[dict], pad_token_id: int = None) -> torch.Tensor:
    """Fait lire un lot d'exemples au modèle en une seule passe et retourne la loss de chaque exemple.

    Paramètres :
        model : modèle 4 bits avec adaptateurs LoRA (sortie de add_lora).
        examples : exemples tokenisés du lot (sortie de tokenize_example).
        pad_token_id : token de remplissage ; inutile (None) pour un lot d'un seul exemple.

    Retour :
        un tenseur [lot] avec la loss de chaque exemple, relié au calcul de gradient.

    Exemple :
        forward_losses(model, examples[:2], pad_token_id=151643)  # tensor([1.2, 0.9], grad_fn=...)
    """
    batch = make_batch(examples, pad_token_id)
    # Pas de labels donnés au modèle : il renverrait la moyenne sur tous les tokens du lot (voir per_example_loss).
    # On lui demande seulement les logits, et on calcule la loss nous-mêmes.
    outputs = model(input_ids=batch["input_ids"].to(model.device), attention_mask=batch["attention_mask"].to(model.device))
    return per_example_loss(outputs.logits, batch["labels"].to(model.device))


def split_into_groups(indices: list[int], batch_size: int, accumulation_steps: int) -> list[list[list[int]]]:
    """Découpe une liste d'indices en lots de batch_size, puis regroupe les lots par accumulation_steps.

    Paramètres :
        indices : numéros d'exemples, déjà mélangés.
        batch_size : nombre d'exemples lus ensemble par le modèle.
        accumulation_steps : nombre de lots dont on additionne les gradients avant une correction.

    Retour :
        une liste de groupes ; un groupe est une liste de lots ; un lot est une liste d'indices.
        Le dernier lot et le dernier groupe peuvent être plus petits.

    Exemple :
        split_into_groups(list(range(5)), 2, 2)  # [[[0, 1], [2, 3]], [[4]]]
    """
    batches = [indices[start:start + batch_size] for start in range(0, len(indices), batch_size)]
    return [batches[start:start + accumulation_steps] for start in range(0, len(batches), accumulation_steps)]


def train_one_group(model, examples: list[dict], group: list[list[int]], pad_token_id: int = None) -> list[float]:
    """Lit les lots d'un groupe et accumule leurs gradients, sans corriger A et B.

    Paramètres :
        model : modèle 4 bits avec adaptateurs LoRA (sortie de add_lora).
        examples : tous les exemples tokenisés.
        group : liste de lots ; un lot est une liste d'indices dans examples (sortie de split_into_groups).
        pad_token_id : token de remplissage (voir forward_losses).

    Retour :
        la loss de chaque exemple du groupe (float, non divisée), dans l'ordre vu.
        Les gradients du groupe sont dans les paramètres (param.grad), prêts pour optimizer.step().

    Exemple :
        train_one_group(model, examples, [[0, 1], [2, 3]], pad_token_id=0)  # 4 loss, gradients accumulés
    """
    # vrai nombre d'exemples du groupe : le dernier groupe peut être plus petit
    group_size = sum(len(batch_indices) for batch_indices in group)

    group_losses = []
    for batch_indices in group:
        # 1. le modèle prédit chaque token ; on obtient la loss de chaque exemple du lot
        #    (moyenne sur les tokens de la réponse du tuteur, les -100 sont ignorés).
        example_losses = forward_losses(model, [examples[index] for index in batch_indices], pad_token_id)
        # 2. backward ADDITIONNE les gradients d'un lot à l'autre. On additionne les loss des exemples
        #    et on divise par le nombre total d'exemples du groupe : chaque exemple pèse 1/group_size,
        #    quelle que soit sa longueur ou la façon de former les lots.
        loss = example_losses.sum() / group_size
        # 3. backward calcule, pour chaque valeur de A et B, comment la loss change si on la modifie un peu
        #    (c'est le gradient : il pointe vers là où la loss AUGMENTE).
        #    backward n'a pas d'argument car il remonte tous les calculs qui ont produit loss
        #    (PyTorch les a enregistrés).
        loss.backward()
        # .tolist() récupère les nombres seuls, sans garder tout le graphe de calcul en mémoire
        group_losses.extend(example_losses.detach().tolist())
    return group_losses


def train_steps(
    model,
    examples: list[dict],
    learning_rate: float = LEARNING_RATE,
    num_epochs: int = 1,
    batch_size: int = 1,
    accumulation_steps: int = ACCUMULATION_STEPS,
    pad_token_id: int = None,
    seed: int = SEED,
    optimizer=None,
    epoch_label: int = None,
    on_optimizer_step=None,
    start_step: int = 0,
) -> tuple[list[float], int]:
    """Entraîne le modèle, par groupes d'exemples, et retourne les loss et le nombre de corrections.

    La loss est le nombre qui mesure l'erreur du modèle : plus il est petit, mieux le modèle prédit.
    Une epoch est un passage complet sur tous les exemples, dans un ordre mélangé.
    Un lot (batch) est une série d'exemples lus ensemble par le modèle, en une seule passe.
    Un groupe est une série de lots dont on additionne les gradients avant une seule correction de A et B.
    Le batch effectif (exemples par correction) vaut batch_size x accumulation_steps.

    ATTENTION mémoire : un lot de 8 exemples longs (jusqu'à ~3 600 tokens) demande beaucoup de mémoire GPU.
    C'est prévu pour un A100 80 Go. Si « CUDA out of memory » : réduire batch_size et augmenter
    accumulation_steps (par exemple 4 x 2 ou 2 x 4) pour garder le même batch effectif.

    Paramètres :
        model : modèle 4 bits avec adaptateurs LoRA (sortie de add_lora).
        examples : exemples déjà tokenisés (sortie de tokenize_example) ; la liste n'est pas modifiée.
        learning_rate : taille du pas de correction.
        num_epochs : nombre de passages sur les exemples.
        batch_size : nombre d'exemples lus ensemble par le modèle (1 : un exemple à la fois, sans padding).
        accumulation_steps : nombre de lots par groupe (le dernier groupe peut être plus petit).
        pad_token_id : token de remplissage, obligatoire si batch_size > 1 (tokenizer.pad_token_id).
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
    # Avec plusieurs exemples par lot, il faut compléter les plus courts : sans token de remplissage, impossible.
    if batch_size > 1 and pad_token_id is None:
        raise ValueError("pad_token_id est obligatoire quand batch_size > 1")

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

        for group in split_into_groups(indices, batch_size, accumulation_steps):
            # Temps 1 à 3 (voir train_one_group) : les gradients du groupe s'accumulent dans A et B.
            # On enregistre les loss NON divisées, sinon les valeurs affichées dépendraient de la taille du groupe.
            group_losses = train_one_group(model, examples, group, pad_token_id)
            group_size = len(group_losses)
            shown_epoch = epoch if epoch_label is None else epoch_label
            for value in group_losses:
                losses.append(value)
                print(f"epoch {shown_epoch}, exemple {len(losses)} : loss {value:.4f}")

            # Temps 4 et 5 : une fois par groupe.
            # 4. step déplace A et B dans le sens OPPOSÉ au gradient (ici la moyenne du groupe),
            #    d'un petit pas, pour faire baisser la loss
            optimizer.step()
            # 5. zero_grad remet les gradients à zéro, sinon ils s'additionneraient d'un groupe à l'autre
            optimizer.zero_grad()
            number_of_optimizer_steps += 1

            if on_optimizer_step is not None:
                # les loss du groupe qui vient d'être corrigé sont les group_size dernières ajoutées à losses
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
            # Un exemple à la fois (pas de padding) et la MÊME fonction que train_steps :
            # l'entraînement et la validation mesurent exactement la même chose.
            total_loss += forward_losses(model, [example]).item()

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


def test_per_example_loss():
    """Test sans GPU : per_example_loss égale la cross_entropy "mean" pour 1 exemple, et pèse chaque exemple pareil."""
    torch.manual_seed(0)
    vocab_size = 7

    # Lot de 1 : le résultat doit être celui de la cross_entropy classique, avec le décalage d'un cran à la main
    logits = torch.randn(1, 5, vocab_size)
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 1, 4]])
    expected = cross_entropy(logits[0, :-1], labels[0, 1:], ignore_index=IGNORE_INDEX)  # reduction="mean" par défaut
    assert torch.allclose(per_example_loss(logits, labels)[0], expected)

    # Lot de 2 exemples de longueurs différentes : A a 2 tokens appris, B en a 5 (A est complété par du padding)
    logits = torch.randn(2, 7, vocab_size)
    labels = torch.tensor([
        [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 2, 5, IGNORE_INDEX, IGNORE_INDEX],
        [IGNORE_INDEX, 1, 4, 0, 3, 6, 2],
    ])
    mean_a = cross_entropy(logits[0, :4], labels[0, 1:5], ignore_index=IGNORE_INDEX)  # A seul, sans padding
    mean_b = cross_entropy(logits[1, :-1], labels[1, 1:], ignore_index=IGNORE_INDEX)
    losses = per_example_loss(logits, labels)
    assert torch.allclose(losses, torch.stack([mean_a, mean_b]))

    # La moyenne sur tous les tokens du lot (ce que donnerait outputs.loss) est DIFFÉRENTE : B y pèserait plus
    token_mean = cross_entropy(logits[:, :-1].reshape(-1, vocab_size), labels[:, 1:].reshape(-1), ignore_index=IGNORE_INDEX)
    assert not torch.allclose(token_mean, losses.mean())

    # Un exemple sans aucun token appris vaut 0 (et pas NaN)
    assert per_example_loss(torch.randn(1, 4, vocab_size), torch.full((1, 4), IGNORE_INDEX)).item() == 0.0
    print(f"per_example_loss : égale la cross_entropy pour 1 exemple ({expected.item():.4f}), "
          f"lot de 2 : {losses.tolist()} (moyenne par exemple {losses.mean().item():.4f} "
          f"contre {token_mean.item():.4f} en moyenne sur tous les tokens)")


def test_split_into_groups():
    """Test sans GPU : 5 exemples en lots de 2 puis groupes de 2 donnent [[0, 1], [2, 3]] et [[4]]."""
    assert split_into_groups(list(range(5)), 2, 2) == [[[0, 1], [2, 3]], [[4]]]
    # batch_size=1 : un lot par exemple, comme avant l'ajout des lots
    assert split_into_groups(list(range(3)), 1, 2) == [[[0], [1]], [[2]]]
    print("split_into_groups : OK")


class TinyModel(torch.nn.Module):
    """Faux modèle pour tester train_one_group sans GPU : un embedding puis une couche linéaire, sans dropout."""

    device = torch.device("cpu")

    def __init__(self, vocab_size: int = 20, hidden_size: int = 8):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask):
        # attention_mask est ignoré : ce faux modèle traite chaque token seul, le padding ne peut rien contaminer
        return SimpleNamespace(logits=self.output(self.embedding(input_ids)))


def test_batch_equivalence():
    """Test sans GPU : 6 exemples en 6x1, 1x6 ou 2x3 (batch x accumulation) donnent les mêmes gradients."""
    torch.manual_seed(0)
    model = TinyModel()

    # 6 exemples de longueurs 4 à 9 ; les k premiers labels sont masqués (-100), le reste est à apprendre
    examples = []
    for length in range(4, 10):
        input_ids = torch.randint(1, 20, (length,)).tolist()
        masked = length // 2
        examples.append({
            "input_ids": input_ids,
            "attention_mask": [1] * length,
            "labels": [IGNORE_INDEX] * masked + input_ids[masked:],
        })

    gradients = {}
    for batch_size, accumulation_steps in ((6, 1), (1, 6), (2, 3)):
        model.zero_grad()
        # indices non mélangés : le même ordre pour les 3 découpages, un seul groupe de 6 exemples
        groups = split_into_groups(list(range(6)), batch_size, accumulation_steps)
        assert len(groups) == 1
        train_one_group(model, examples, groups[0], pad_token_id=0)
        gradients[(batch_size, accumulation_steps)] = [param.grad.clone() for param in model.parameters()]

    reference = gradients[(6, 1)]
    for key in ((1, 6), (2, 3)):
        for expected, obtained in zip(reference, gradients[key]):
            assert torch.allclose(expected, obtained, atol=1e-6), f"gradients différents pour le découpage {key}"
    print("test_batch_equivalence : 6x1, 1x6 et 2x3 donnent les mêmes gradients (atol 1e-6)")


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


def test_batch_steps():
    """Test : 5 exemples en lots de 2, groupes de 2 lots, donnent 2 corrections (4 + 1 exemples) et 5 loss."""
    model, tokenizer = load_model_4bit()
    model = add_lora(model)

    conversations = load_split("train")[21:26]
    examples = [tokenize_example(messages, tokenizer) for messages in conversations]

    # le token de remplissage de Qwen existe (<|endoftext|>) ; on prend eos au cas où un autre modèle n'en aurait pas
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    losses, number_of_optimizer_steps = train_steps(
        model, examples, num_epochs=1, batch_size=2, accumulation_steps=2, pad_token_id=pad_token_id,
    )
    # lots : 2 + 2 + 1 ; groupes : (2 + 2) et (1) ; donc 2 corrections
    assert number_of_optimizer_steps == 2
    assert len(losses) == 5
    print("lots : 5 exemples, lots de 2, groupes de 2 lots → 2 corrections")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Lancer depuis la racine du dépôt : python -m src.training.train
    # Avec --self-test : seulement les tests sans modèle ni GPU (les 3 premiers ci-dessous).
    test_per_example_loss()
    test_split_into_groups()
    test_batch_equivalence()
    if "--self-test" in sys.argv:
        raise SystemExit(0)
    # L'ordre compte : test_save_and_reload lit l'adaptateur écrit par test_memorize_small_batch.
    test_memorize_small_batch()
    test_save_and_reload()
    test_accumulation()
    test_callback_and_stop()
    test_batch_steps()
