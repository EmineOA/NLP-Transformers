"""Entraînement complet QLoRA (bloc C).

Ce fichier lance le vrai entraînement : le modèle voit tous les exemples train, epoch après epoch.
Une epoch est un passage complet sur les exemples. Un pas d'optimisation est une correction de A et B
(une correction tous les batch_size x accumulation_steps exemples, 8 par défaut).
Après chaque epoch, et aussi tous les N pas avec --eval-every N, on mesure la loss de validation
(l'erreur sur des exemples jamais vus).
Chaque fois qu'elle bat son record, l'adaptateur LoRA est sauvegardé dans {output_dir}/best.
Les loss sont écrites dans des fichiers CSV au fur et à mesure, pour survivre à une coupure :
historique_pas.csv (loss train de chaque pas), historique_val.csv (chaque mesure de loss val)
et historique_loss.csv (une ligne par epoch).
Les tests de la boucle sont dans train.py ; ceux des petites fonctions de ce fichier : --self-test.
"""

# Lancer depuis la racine : python -m src.training.run_training (~4 h par epoch sur RTX 4070 12 Go, mémoire saturée ; bien plus rapide sur Colab A100, à mesurer ; fermer les jeux)
# Test rapide : python -m src.training.run_training --quick
# Vrai batch (A100 80 Go) : python -m src.training.run_training --batch-size 4 --accumulation-steps 2
# Test sans GPU des petites fonctions : python -m src.training.run_training --self-test

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import torch

from src.llm.model import DEFAULT_MODEL_NAME
from src.training.dataset import MAX_LENGTH, load_split, tokenize_example
from src.training.model import LORA_ALPHA, LORA_DROPOUT, LORA_RANK, LORA_TARGET_MODULES, add_lora, load_model_4bit
from src.training.train import ACCUMULATION_STEPS, LEARNING_RATE, SEED, evaluate_loss, train_steps

# Mode rapide : pour vérifier le script en ~2 minutes avant le vrai lancement.
QUICK_TRAIN_EXAMPLES = 16
QUICK_VAL_EXAMPLES = 8
QUICK_EPOCHS = 2
# 2 corrections par epoch en mode rapide : mesurer tous les 2 pas teste le chemin "mesure en cours d'epoch".
QUICK_EVAL_EVERY = 2

# v0.2 : version de l'adaptateur ; dossier ignoré par git.
OUTPUT_DIR = "adapters/v0.2"
QUICK_OUTPUT_DIR = "adapters/quick"

# Fichiers qui prouvent qu'un dossier contient déjà un entraînement (les dossiers epoch_* sont cherchés à part).
USED_FILE_NAMES = ("best", "best_tmp", "config.json", "historique_pas.csv", "historique_val.csv", "historique_loss.csv", "resultat.json")

# Raisons d'arrêt écrites dans resultat.json.
STOP_EARLY = "arrêt anticipé"
STOP_EPOCHS = "nombre d'epochs atteint"


def append_csv_row(csv_path, row: list) -> None:
    """Ajoute une ligne à la fin d'un fichier CSV, tout de suite sur le disque.

    Paramètres :
        csv_path : chemin du fichier (créé s'il n'existe pas).
        row : valeurs de la ligne, dans l'ordre des colonnes.

    Retour :
        rien (le fichier est écrit).

    Exemple :
        append_csv_row("adapters/v0.2/historique_pas.csv", [1, 1, "1.5000"])
    """
    # Ouvrir et fermer à chaque ligne : si la session plante, tout ce qui a été écrit est déjà sur le disque.
    # flush + fsync : sur Google Drive (Colab), sans eux la ligne peut rester en mémoire et se perdre à la coupure.
    with open(csv_path, "a", encoding="utf-8", newline="") as file:
        csv.writer(file).writerow(row)
        file.flush()
        os.fsync(file.fileno())


def new_state() -> dict:
    """Crée le carnet de suivi de l'entraînement (meilleure loss val, patience, dernière mesure).

    Retour :
        un dict avec epoch (epoch en cours), best_loss_val / best_step / best_epoch (le record),
        evaluations_without_improvement (mesures val de suite sans record), last_eval_step,
        last_loss_val (dernière mesure) et stopped (True quand la patience est épuisée).

    Exemple :
        new_state()["stopped"]  # False
    """
    return {
        "epoch": 1,
        "best_loss_val": None,
        "best_step": None,
        "best_epoch": None,
        "evaluations_without_improvement": 0,
        "last_eval_step": None,
        "last_loss_val": None,
        "stopped": False,
    }


def update_best_state(state: dict, loss_val: float, step: int, epoch: int, patience: int) -> bool:
    """Note une mesure de loss val dans le carnet et dit si c'est un nouveau record.

    Paramètres :
        state : carnet retourné par new_state (modifié sur place).
        loss_val : loss de validation qui vient d'être mesurée.
        step : numéro du pas d'optimisation de la mesure.
        epoch : epoch en cours.
        patience : nombre de mesures de suite sans record qui arrêtent l'entraînement (0 : jamais).

    Retour :
        True si loss_val bat le record (l'appelant doit alors sauvegarder l'adaptateur).

    Exemple :
        update_best_state(new_state(), 1.5, step=8, epoch=1, patience=2)  # True
    """
    state["last_eval_step"] = step
    state["last_loss_val"] = loss_val
    if state["best_loss_val"] is None or loss_val < state["best_loss_val"]:
        state["best_loss_val"] = loss_val
        state["best_step"] = step
        state["best_epoch"] = epoch
        state["evaluations_without_improvement"] = 0
        return True
    state["evaluations_without_improvement"] += 1
    if patience > 0 and state["evaluations_without_improvement"] >= patience:
        state["stopped"] = True
    return False


def save_best(model, output_dir: str, state: dict) -> None:
    """Sauvegarde l'adaptateur dans {output_dir}/best (remplace l'ancien) et note d'où il vient.

    Paramètres :
        model : modèle 4 bits avec adaptateurs LoRA (ou tout objet qui a save_pretrained).
        output_dir : dossier de l'entraînement.
        state : carnet dont best_step, best_epoch et best_loss_val viennent d'être mis à jour.

    Retour :
        rien (best/meilleur_pas.txt et best/meilleure_epoch.txt sont écrits à côté de l'adaptateur).

    Exemple :
        save_best(model, "adapters/v0.2", state)  # adapters/v0.2/best/meilleur_pas.txt : "pas=8"...
    """
    best_dir = Path(output_dir) / "best"
    tmp_dir = Path(output_dir) / "best_tmp"
    # Reste d'une coupure pendant une sauvegarde précédente : on repart d'un dossier vide.
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    # On écrit tout dans best_tmp d'abord : une coupure de session Colab pendant l'écriture
    # ne touche pas à best/, qui garde le dernier adaptateur complet.
    # save_pretrained écrit seulement A et B (quelques dizaines de Mo) : l'appeler à chaque record reste rapide.
    model.save_pretrained(str(tmp_dir))
    with open(tmp_dir / "meilleur_pas.txt", "w", encoding="utf-8") as file:
        file.write(f"pas={state['best_step']}\nepoch={state['best_epoch']}\nloss_val={state['best_loss_val']:.4f}\n")
    # meilleure_epoch.txt existait déjà dans les entraînements précédents : on le garde pour ne rien casser
    with open(tmp_dir / "meilleure_epoch.txt", "w", encoding="utf-8") as file:
        file.write(f"{state['best_epoch']}\n")

    # Tout est écrit : on remplace l'ancien best/ seulement maintenant.
    # (os.replace ne remplace pas un dossier non vide sous Linux, d'où le rmtree juste avant.)
    if best_dir.exists():
        shutil.rmtree(best_dir)
    os.replace(tmp_dir, best_dir)


def evaluate_and_log(model, val_examples: list[dict], output_dir: str, state: dict, patience: int, step: int) -> None:
    """Mesure la loss val sur tout val, l'écrit dans historique_val.csv et sauvegarde si c'est un record.

    Paramètres :
        model : modèle 4 bits avec adaptateurs LoRA.
        val_examples : tous les exemples val tokenisés.
        output_dir : dossier de l'entraînement.
        state : carnet de suivi (modifié sur place ; state["epoch"] donne l'epoch en cours).
        patience : voir update_best_state.
        step : numéro du pas d'optimisation de la mesure.

    Retour :
        rien (state["stopped"] passe à True si la patience est épuisée).

    Exemple :
        evaluate_and_log(model, val_examples, "adapters/v0.2", state, patience=2, step=8)
    """
    loss_val = evaluate_loss(model, val_examples)
    append_csv_row(Path(output_dir) / "historique_val.csv", [step, state["epoch"], f"{loss_val:.4f}"])
    is_best = update_best_state(state, loss_val, step, state["epoch"], patience)
    print(f"pas {step} : loss val {loss_val:.4f}{' (meilleure)' if is_best else ''}")
    if is_best:
        save_best(model, output_dir, state)
    if state["stopped"]:
        print(f"arrêt anticipé : la loss val ne s'améliore plus depuis {state['evaluations_without_improvement']} mesure(s)")


def make_step_callback(model, val_examples: list[dict], output_dir: str, eval_every: int, patience: int, state: dict):
    """Crée la fonction que train_steps appelle après chaque correction de A et B.

    Paramètres :
        model : modèle 4 bits avec adaptateurs LoRA.
        val_examples : tous les exemples val tokenisés.
        output_dir : dossier de l'entraînement.
        eval_every : mesure val tous les eval_every pas (0 : jamais en cours d'epoch).
        patience : voir update_best_state.
        state : carnet de suivi (modifié sur place).

    Retour :
        une fonction on_step(step, group_loss) qui renvoie True quand il faut arrêter l'entraînement.

    Exemple :
        on_step = make_step_callback(model, val_examples, "adapters/v0.2", 8, 3, new_state())
    """

    def on_step(step: int, group_loss: float) -> bool:
        append_csv_row(Path(output_dir) / "historique_pas.csv", [step, state["epoch"], f"{group_loss:.4f}"])
        if eval_every > 0 and step % eval_every == 0:
            evaluate_and_log(model, val_examples, output_dir, state, patience, step)
        return state["stopped"]

    return on_step


def build_config(
    num_epochs: int,
    learning_rate: float,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    patience: int,
    eval_every: int,
    num_train_examples: int,
    num_val_examples: int,
    gpu_name: str,
    batch_size: int = 1,
    accumulation_steps: int = ACCUMULATION_STEPS,
) -> dict:
    """Rassemble les réglages de l'entraînement, tels qu'ils seront écrits dans config.json.

    Paramètres :
        num_epochs, learning_rate, lora_rank, lora_alpha, lora_dropout, patience, eval_every : réglages de run_training.
        num_train_examples, num_val_examples : nombre d'exemples réellement utilisés.
        gpu_name : nom de la carte graphique.
        batch_size : nombre d'exemples lus ensemble par le modèle.
        accumulation_steps : nombre de lots dont on additionne les gradients avant une correction.

    Retour :
        un dict prêt pour json.dump (batch_size_effectif = batch_size_par_pas x accumulation_steps ;
        quantification et max_length décrivent les constantes du projet).

    Exemple :
        build_config(3, 2e-4, 16, 32, 0.05, 0, 0, 885, 118, "RTX 4070")["batch_size_effectif"]  # 8
    """
    return {
        "model": DEFAULT_MODEL_NAME,
        "quantification": "4 bits nf4, double quantization, calcul bfloat16/float16",
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": LORA_TARGET_MODULES,
        "learning_rate": learning_rate,
        "batch_size_par_pas": batch_size,
        "accumulation_steps": accumulation_steps,
        "batch_size_effectif": batch_size * accumulation_steps,
        "max_length": MAX_LENGTH,
        "epochs": num_epochs,
        "patience": patience,
        "eval_every": eval_every,
        "num_train_examples": num_train_examples,
        "num_val_examples": num_val_examples,
        "gpu": gpu_name,
    }


def epoch_is_complete(steps_done: int, number_of_examples: int, batch_size: int, accumulation_steps: int) -> bool:
    """Dit si une epoch est allée jusqu'au bout (tous les groupes d'exemples ont été corrigés).

    Paramètres :
        steps_done : nombre de corrections faites pendant l'epoch.
        number_of_examples : nombre d'exemples train.
        batch_size : nombre d'exemples par lot.
        accumulation_steps : nombre de lots par groupe.

    Retour :
        True si steps_done vaut le nombre de groupes de l'epoch (le dernier groupe peut être plus petit).

    Exemple :
        epoch_is_complete(3, 10, 1, 4)  # True : 10 exemples en groupes de 4 = 3 groupes
        epoch_is_complete(2, 10, 2, 2)  # False : 4 exemples par groupe, il en faut 3
    """
    # un groupe contient batch_size x accumulation_steps exemples (le dernier peut en avoir moins)
    number_of_groups = math.ceil(number_of_examples / (batch_size * accumulation_steps))
    return steps_done == number_of_groups


def run_one_epoch(
    model, train_examples, val_examples, optimizer, epoch, step_before, output_dir, eval_every, patience, state,
    batch_size=1, accumulation_steps=ACCUMULATION_STEPS, pad_token_id=None,
):
    """Entraîne une epoch, mesure la loss val à sa fin, sauvegarde epoch_N et écrit historique_loss.csv.

    Si l'entraînement s'arrête au milieu de l'epoch, rien n'est écrit dans historique_loss.csv
    ni dans epoch_N : ce fichier ne contient que des epochs complètes.

    Paramètres :
        model, train_examples, val_examples, optimizer : voir run_training.
        epoch : numéro de l'epoch (à partir de 1).
        step_before : nombre de pas d'optimisation déjà faits avant cette epoch.
        output_dir, eval_every, patience : voir run_training.
        state : carnet de suivi (modifié sur place).
        batch_size, accumulation_steps, pad_token_id : voir train_steps.

    Retour :
        {"epoch", "loss_train", "loss_val", "seconds", "adapter_dir", "steps", "complete"} ;
        complete vaut False si l'entraînement a été arrêté au milieu de l'epoch (adapter_dir vaut alors None).

    Exemple :
        result = run_one_epoch(model, train, val, optimizer, 1, 0, "adapters/v0.2", 8, 3, new_state())
    """
    start_time = time.perf_counter()
    state["epoch"] = epoch
    on_step = make_step_callback(model, val_examples, output_dir, eval_every, patience, state)

    # Seed différente à chaque epoch : train_steps repart de son epoch 1 à chaque appel ;
    # sans seed différente, l'ordre de mélange serait le même à chaque epoch.
    losses, steps = train_steps(
        model, train_examples, num_epochs=1, batch_size=batch_size, accumulation_steps=accumulation_steps,
        pad_token_id=pad_token_id, optimizer=optimizer, seed=SEED + epoch, epoch_label=epoch,
        on_optimizer_step=on_step, start_step=step_before,
    )
    step_after = step_before + steps
    # Si une mesure vient d'avoir lieu à ce pas (eval_every tombe pile en fin d'epoch), on ne la refait pas.
    if state["last_eval_step"] != step_after:
        evaluate_and_log(model, val_examples, output_dir, state, patience, step_after)

    # loss train = moyenne pendant l'epoch (le modèle s'améliore en cours de route) ;
    # loss val = mesurée à la fin de l'epoch : les deux ne sont pas exactement comparables.
    loss_train = sum(losses) / len(losses)
    loss_val = state["last_loss_val"]
    complete = epoch_is_complete(steps, len(train_examples), batch_size, accumulation_steps)
    adapter_dir = None

    if complete:
        adapter_dir = f"{output_dir}/epoch_{epoch}"
        model.save_pretrained(adapter_dir)
        seconds = time.perf_counter() - start_time
        append_csv_row(Path(output_dir) / "historique_loss.csv", [epoch, f"{loss_train:.4f}", f"{loss_val:.4f}", f"{seconds:.0f}"])
        print(f"epoch {epoch} : loss train {loss_train:.4f}, loss val {loss_val:.4f}, {seconds:.0f} s")
    else:
        seconds = time.perf_counter() - start_time
        print(f"epoch {epoch} interrompue au pas {step_after} ({steps} pas faits dans cette epoch)")
    return {
        "epoch": epoch, "loss_train": loss_train, "loss_val": loss_val,
        "seconds": seconds, "adapter_dir": adapter_dir, "steps": steps, "complete": complete,
    }


def write_result(output_dir: str, state: dict, total_seconds: float, total_steps: int, last_epoch: dict) -> dict:
    """Écrit resultat.json : meilleur pas, meilleure epoch, meilleure loss val, raison de l'arrêt, durée.

    Paramètres :
        output_dir : dossier de l'entraînement.
        state : carnet de suivi à la fin de l'entraînement.
        total_seconds : durée totale de l'entraînement (chargement du modèle exclu).
        total_steps : nombre de pas d'optimisation faits.
        last_epoch : dict retourné par run_one_epoch pour la dernière epoch lancée ;
            il dit si cette epoch a été interrompue (epoch_partielle) et combien de pas elle a faits (pas_dans_epoch).

    Retour :
        le dict écrit dans le fichier.

    Exemple :
        write_result("adapters/v0.2", state, 60.0, 4, {"complete": True, "steps": 2})["raison_arret"]
        # "nombre d'epochs atteint"
    """
    result = {
        "meilleur_pas": state["best_step"],
        "meilleure_epoch": state["best_epoch"],
        "meilleure_loss_val": round(state["best_loss_val"], 4),
        "raison_arret": STOP_EARLY if state["stopped"] else STOP_EPOCHS,
        "pas_total": total_steps,
        "epoch_partielle": not last_epoch["complete"],
        "pas_dans_epoch": last_epoch["steps"],
        "duree_totale_secondes": round(total_seconds),
    }
    with open(Path(output_dir) / "resultat.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    return result


def prepare_output_dir(output_dir: str, quick: bool) -> str:
    """Choisit le dossier de sortie et refuse de démarrer s'il contient déjà un entraînement.

    Paramètres :
        output_dir : dossier demandé (None : adapters/v0.2, ou adapters/quick en mode quick).
        quick : en mode quick, le dossier est supprimé puis recréé : c'est un test jetable.

    Retour :
        le dossier à utiliser (str).

    Exemple :
        prepare_output_dir(None, quick=True)  # "adapters/quick"
    """
    if output_dir is None:
        output_dir = QUICK_OUTPUT_DIR if quick else OUTPUT_DIR
    if quick:
        if Path(output_dir).exists():
            shutil.rmtree(output_dir)
        return output_dir

    # Un entraînement complet coûte plusieurs heures (~4 h par epoch sur RTX 4070) : on refuse de l'écraser.
    # Les CSV comptent aussi : relancer dans un dossier planté tôt ajouterait un 2e en-tête et casserait la lecture.
    # Ce test est fait AVANT de charger le modèle, pour échouer tout de suite.
    folder = Path(output_dir)
    leftovers = [name for name in USED_FILE_NAMES if (folder / name).exists()] + [path.name for path in folder.glob("epoch_*")]
    if leftovers:
        raise SystemExit(
            f"{output_dir} contient déjà un entraînement ({', '.join(leftovers)}) : supprime-le ou change de dossier"
        )
    return output_dir


def prepare_data(quick: bool, max_train: int, tokenizer) -> tuple[list[dict], list[dict]]:
    """Charge les conversations train et val et les tokenise.

    Paramètres :
        quick : True pour ne garder que 16 exemples train et 8 val.
        max_train : nombre maximal d'exemples train (0 : tous) ; ignoré en mode quick.
        tokenizer : tokenizer du modèle.

    Retour :
        (train_examples, val_examples), des exemples déjà tokenisés (sortie de tokenize_example).

    Exemple :
        train_examples, val_examples = prepare_data(True, 0, tokenizer)  # 16 et 8 exemples
    """
    train_conversations = load_split("train")
    val_conversations = load_split("val")
    if quick:
        train_conversations = train_conversations[:QUICK_TRAIN_EXAMPLES]
        val_conversations = val_conversations[:QUICK_VAL_EXAMPLES]
    elif max_train > 0:
        train_conversations = train_conversations[:max_train]
    train_examples = [tokenize_example(messages, tokenizer) for messages in train_conversations]
    val_examples = [tokenize_example(messages, tokenizer) for messages in val_conversations]
    return train_examples, val_examples


def init_output_dir(output_dir: str, config: dict) -> None:
    """Crée le dossier de sortie, les 3 CSV (avec leur en-tête) et config.json.

    Paramètres :
        output_dir : dossier de l'entraînement.
        config : réglages, sortie de build_config.

    Retour :
        rien (les fichiers sont écrits).

    Exemple :
        init_output_dir("adapters/v0.2", build_config(3, 2e-4, 16, 32, 0.05, 0, 0, 885, 118, "RTX 4070"))
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    append_csv_row(Path(output_dir) / "historique_loss.csv", ["epoch", "loss_train", "loss_val", "secondes"])
    append_csv_row(Path(output_dir) / "historique_pas.csv", ["pas", "epoch", "loss_train"])
    append_csv_row(Path(output_dir) / "historique_val.csv", ["pas", "epoch", "loss_val"])
    # Les réglages sont écrits AVANT l'entraînement : même si la session plante, on sait avec quoi elle tournait.
    with open(Path(output_dir) / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)


def run_training(
    num_epochs: int = 3,
    quick: bool = False,
    output_dir: str = None,
    learning_rate: float = LEARNING_RATE,
    lora_rank: int = LORA_RANK,
    lora_alpha: int = LORA_ALPHA,
    lora_dropout: float = LORA_DROPOUT,
    patience: int = 0,
    max_train: int = 0,
    eval_every: int = None,
    batch_size: int = 1,
    accumulation_steps: int = ACCUMULATION_STEPS,
) -> list[dict]:
    """Entraîne le modèle epoch par epoch, sauvegarde chaque epoch et retourne les résultats.

    Paramètres :
        num_epochs : nombre d'epochs (ignoré en mode quick, qui en fait toujours 2).
        quick : True pour un test rapide (16 exemples train, 8 val, 2 epochs, dossier adapters/quick).
        output_dir : dossier de sortie (None : adapters/v0.2, ou adapters/quick en mode quick).
            Le programme refuse de démarrer si ce dossier contient déjà un entraînement (sauf en mode quick).
        learning_rate : taille du pas de correction.
        lora_rank : r de LoRA (taille des petites matrices A et B).
        lora_alpha : alpha de LoRA (force de la correction).
        lora_dropout : dropout de LoRA (proportion de valeurs mises à 0 pendant l'entraînement).
        patience : arrêt anticipé après `patience` mesures de loss val de suite sans record
            (0 : désactivé, on fait toutes les epochs). Avec eval_every=0, une mesure = une epoch.
        max_train : nombre maximal d'exemples train utilisés (0 : tous), pour tester vite ; ignoré en mode quick.
        eval_every : mesure la loss val sur tout val tous les eval_every pas d'optimisation, en plus de la fin
            de chaque epoch (None ou 0 : fin d'epoch seulement ; None en mode quick : tous les 2 pas).
        batch_size : nombre d'exemples lus ensemble par le modèle en une passe (1 : un exemple à la fois).
        accumulation_steps : nombre de lots dont on additionne les gradients avant une correction.
            Batch effectif = batch_size x accumulation_steps (8 par défaut). Un lot de 8 exemples longs
            (jusqu'à ~3 600 tokens) demande beaucoup de mémoire GPU : prévu pour un A100 80 Go.
            Si « CUDA out of memory », réduire batch_size et augmenter accumulation_steps (4 x 2, 2 x 4...).

    Retour :
        une liste de dicts, un par epoch : {"epoch", "loss_train", "loss_val", "seconds", "adapter_dir", "steps"}.
        Seules les epochs complètes y figurent (une epoch interrompue par l'arrêt anticipé est décrite dans resultat.json).
        Fichiers écrits dans output_dir : config.json (réglages), resultat.json (meilleur pas, raison d'arrêt, durée,
        epoch_partielle, pas_dans_epoch), historique_pas.csv, historique_val.csv, historique_loss.csv, epoch_N/
        et best/ (meilleur adaptateur selon la loss val, avec meilleur_pas.txt).

    Exemple :
        results = run_training(quick=True)
        len(results)  # 2
    """
    if batch_size < 1 or accumulation_steps < 1:
        raise SystemExit("--batch-size et --accumulation-steps doivent valoir au moins 1")
    if eval_every is None:
        eval_every = QUICK_EVAL_EVERY if quick else 0
    if quick:
        num_epochs = QUICK_EPOCHS
    output_dir = prepare_output_dir(output_dir, quick)

    model, tokenizer = load_model_4bit()
    # Token de remplissage du padding. Qwen en a un (<|endoftext|>) ; eos en secours si un autre modèle n'en avait pas.
    # Sa valeur importe peu : attention_mask et labels l'ignorent (voir make_batch).
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model = add_lora(model, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
    train_examples, val_examples = prepare_data(quick, max_train, tokenizer)
    print(f"{len(train_examples)} exemples train, {len(val_examples)} exemples val, {num_epochs} epochs")
    print(f"batch {batch_size} x accumulation {accumulation_steps} = {batch_size * accumulation_steps} exemples par pas")

    # Un seul optimiseur pour tout l'entraînement : il garde sa mémoire d'une epoch à l'autre.
    # Il ne reçoit que les paramètres entraînables (A et B).
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

    config = build_config(
        num_epochs, learning_rate, lora_rank, lora_alpha, lora_dropout, patience, eval_every,
        len(train_examples), len(val_examples), torch.cuda.get_device_name(0), batch_size, accumulation_steps,
    )
    init_output_dir(output_dir, config)
    torch.cuda.reset_peak_memory_stats()

    state = new_state()
    results = []
    total_steps = 0
    start_time = time.perf_counter()
    for epoch in range(1, num_epochs + 1):
        result = run_one_epoch(
            model, train_examples, val_examples, optimizer, epoch, total_steps, output_dir, eval_every, patience, state,
            batch_size, accumulation_steps, pad_token_id,
        )
        total_steps += result["steps"]
        # une epoch interrompue n'est pas un résultat d'epoch : son pas et sa raison vont dans resultat.json
        if result["complete"]:
            results.append(result)
        # Arrêt anticipé : si la loss val ne baisse plus, continuer ne fait que surapprendre
        # et coûte des heures pour rien (ex. v0.2 : loss val 1.476 puis 1.482 → arrêt après l'epoch 2).
        if state["stopped"]:
            break

    # best/ contient déjà l'adaptateur du dernier record (écrit pendant l'entraînement) :
    # le reste du projet (generate_response, Massi) charge toujours {output_dir}/best.
    summary = write_result(output_dir, state, time.perf_counter() - start_time, total_steps, result)
    print(f"meilleur adaptateur : pas {summary['meilleur_pas']}, epoch {summary['meilleure_epoch']}, "
          f"loss val {summary['meilleure_loss_val']} → {output_dir}/best ({summary['raison_arret']})")

    peak_memory_in_gb = torch.cuda.max_memory_allocated() / 1024**3  # octets -> Go
    print(f"mémoire GPU maximale : {peak_memory_in_gb:.2f} Go")
    return results


class FakeModel:
    """Faux modèle pour tester save_best sans GPU : save_pretrained écrit un petit fichier."""

    def __init__(self, content: str):
        self.content = content

    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_model.safetensors").write_text(self.content, encoding="utf-8")


def test_save_best_and_output_dir() -> None:
    """Test sans GPU : save_best remplace best/ sans laisser best_tmp, et prepare_output_dir refuse un dossier utilisé."""
    with tempfile.TemporaryDirectory() as folder:
        state = {"best_step": 8, "best_epoch": 1, "best_loss_val": 1.5}
        # un best_tmp resté d'une coupure précédente doit être nettoyé
        (Path(folder) / "best_tmp").mkdir()
        (Path(folder) / "best_tmp" / "reste.txt").write_text("coupure", encoding="utf-8")
        save_best(FakeModel("premier"), folder, state)
        state.update(best_step=16, best_epoch=2, best_loss_val=1.4)
        save_best(FakeModel("second"), folder, state)

        best_dir = Path(folder) / "best"
        assert not (Path(folder) / "best_tmp").exists()
        assert (best_dir / "adapter_model.safetensors").read_text(encoding="utf-8") == "second"
        assert (best_dir / "meilleur_pas.txt").read_text(encoding="utf-8") == "pas=16\nepoch=2\nloss_val=1.4000\n"
        assert (best_dir / "meilleure_epoch.txt").read_text(encoding="utf-8") == "2\n"
        assert not (best_dir / "reste.txt").exists()

        # le dossier contient best/ : un lancement normal doit être refusé, avec un message en français
        try:
            prepare_output_dir(folder, quick=False)
            raise AssertionError("le dossier utilisé aurait dû être refusé")
        except SystemExit as error:
            assert "contient déjà un entraînement" in str(error)
        # un dossier vide (ou absent) est accepté
        assert prepare_output_dir(str(Path(folder) / "neuf"), quick=False) == str(Path(folder) / "neuf")


def self_test() -> None:
    """Test sans GPU : carnet de suivi, patience, écriture CSV et config."""
    # record, puis 2 mesures sans record : avec patience=2, la 2e déclenche l'arrêt
    state = new_state()
    assert update_best_state(state, 1.50, step=8, epoch=1, patience=2) is True
    assert update_best_state(state, 1.60, step=16, epoch=1, patience=2) is False and not state["stopped"]
    assert update_best_state(state, 1.55, step=24, epoch=2, patience=2) is False and state["stopped"]
    assert (state["best_step"], state["best_epoch"], state["last_loss_val"]) == (8, 1, 1.55)
    # patience=0 : jamais d'arrêt, même sans record
    state = new_state()
    for step, loss in ((1, 1.0), (2, 2.0), (3, 3.0)):
        update_best_state(state, loss, step, epoch=1, patience=0)
    assert not state["stopped"] and state["evaluations_without_improvement"] == 2

    with tempfile.TemporaryDirectory() as folder:
        csv_path = Path(folder) / "test.csv"
        append_csv_row(csv_path, ["pas", "loss"])
        append_csv_row(csv_path, [1, "1.5000"])
        assert csv_path.read_text(encoding="utf-8").splitlines() == ["pas,loss", "1,1.5000"]
        state["best_step"], state["best_epoch"], state["best_loss_val"] = 3, 1, 1.0
        assert write_result(folder, state, 12.4, 3, {"complete": False, "steps": 1})["raison_arret"] == STOP_EPOCHS
        saved = json.loads((Path(folder) / "resultat.json").read_text(encoding="utf-8"))
        assert saved["duree_totale_secondes"] == 12 and saved["epoch_partielle"] is True and saved["pas_dans_epoch"] == 1

    # 10 exemples en groupes de 4 : 3 groupes
    assert epoch_is_complete(3, 10, 1, 4) and not epoch_is_complete(2, 10, 1, 4)
    # avec des lots de 2 et 2 lots par groupe : 4 exemples par groupe, donc encore 3 groupes
    assert epoch_is_complete(3, 10, 2, 2) and not epoch_is_complete(2, 10, 2, 2)
    # 16 exemples (mode quick), lots de 2, 2 lots par groupe : 4 corrections par epoch
    assert epoch_is_complete(4, 16, 2, 2) and not epoch_is_complete(3, 16, 2, 2)
    test_save_best_and_output_dir()

    config = build_config(3, 2e-4, 16, 32, 0.05, 0, 10, 885, 118, "gpu de test")
    assert config["batch_size_effectif"] == config["accumulation_steps"] == 8
    assert config["max_length"] == MAX_LENGTH and config["eval_every"] == 10
    config = build_config(3, 2e-4, 16, 32, 0.05, 0, 10, 885, 118, "gpu de test", batch_size=4, accumulation_steps=2)
    assert (config["batch_size_par_pas"], config["accumulation_steps"], config["batch_size_effectif"]) == (4, 2, 8)
    print("self-test run_training : OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement complet QLoRA du tuteur.")
    parser.add_argument("--epochs", type=int, default=3, help="nombre d'epochs (ignoré avec --quick, qui en fait 2)")
    parser.add_argument("--quick", action="store_true", help="test rapide : 16 exemples train, 8 val, 2 epochs")
    parser.add_argument("--self-test", action="store_true", help="teste les petites fonctions sans GPU, puis quitte")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="taille du pas de correction")
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK, help="r de LoRA : taille des petites matrices")
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA, help="alpha de LoRA : force de la correction")
    parser.add_argument("--lora-dropout", type=float, default=LORA_DROPOUT, help="dropout de LoRA (0.05 = 5 %%)")
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="arrêt anticipé après N mesures de loss val de suite sans record (0 : désactivé ; "
        "sans --eval-every, une mesure = une epoch)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="mesure la loss val sur tout val tous les N pas d'optimisation, en plus de la fin d'epoch "
        "(0 : fin d'epoch seulement, défaut ; avec --quick : 2)",
    )
    parser.add_argument("--output-dir", default=None, help=f"dossier de sortie (défaut : {OUTPUT_DIR})")
    parser.add_argument("--max-train", type=int, default=0, help="nombre maximal d'exemples train (0 : tous), pour tester vite")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="exemples lus ensemble par le modèle (défaut 1). Un lot de 8 exemples longs demande beaucoup de "
        "mémoire GPU (prévu pour un A100 80 Go) ; si CUDA out of memory, réduire --batch-size et "
        "augmenter --accumulation-steps",
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=ACCUMULATION_STEPS,
        help=f"lots dont on additionne les gradients avant une correction (défaut {ACCUMULATION_STEPS}) ; "
        "batch effectif = --batch-size x --accumulation-steps",
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
        raise SystemExit(0)
    run_training(
        num_epochs=args.epochs,
        quick=args.quick,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        patience=args.patience,
        max_train=args.max_train,
        eval_every=args.eval_every,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
    )
