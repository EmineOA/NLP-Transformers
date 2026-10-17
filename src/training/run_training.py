"""Entraînement complet QLoRA (bloc C).

Ce fichier lance le vrai entraînement : le modèle voit tous les exemples train, epoch après epoch.
Une epoch est un passage complet sur les exemples. Après chaque epoch, on mesure la loss de validation
(l'erreur sur des exemples jamais vus) et on sauvegarde l'adaptateur LoRA dans son propre dossier.
À la fin, on retient l'epoch dont la loss de validation est la plus basse.
Les tests de la boucle sont dans train.py.
"""

# Lancer depuis la racine : python -m src.training.run_training (environ 45 min par epoch sur RTX 4070 ; fermer les jeux)
# Test rapide : python -m src.training.run_training --quick

import argparse
import csv
import shutil
import time
from pathlib import Path

import torch

from src.training.dataset import load_split, tokenize_example
from src.training.model import add_lora, load_model_4bit
from src.training.train import LEARNING_RATE, SEED, evaluate_loss, train_steps

# Mode rapide : pour vérifier le script en ~2 minutes avant le vrai lancement.
QUICK_TRAIN_EXAMPLES = 16
QUICK_VAL_EXAMPLES = 8
QUICK_EPOCHS = 2

# v0.2 : version de l'adaptateur ; dossier ignoré par git.
OUTPUT_DIR = "adapters/v0.2"
QUICK_OUTPUT_DIR = "adapters/quick"


def run_training(num_epochs: int = 3, quick: bool = False, output_dir: str = None) -> list[dict]:
    """Entraîne le modèle epoch par epoch, sauvegarde chaque epoch et retourne les résultats.

    Paramètres :
        num_epochs : nombre d'epochs (ignoré en mode quick, qui en fait toujours 2).
        quick : True pour un test rapide (16 exemples train, 8 val, 2 epochs, dossier adapters/quick).
        output_dir : dossier de sortie (None : adapters/v0.2, ou adapters/quick en mode quick).

    Retour :
        une liste de dicts, un par epoch : {"epoch", "loss_train", "loss_val", "seconds", "adapter_dir"}.
        Le meilleur adaptateur est aussi copié dans {output_dir}/best.

    Exemple :
        results = run_training(quick=True)
        len(results)  # 2
    """
    if quick:
        num_epochs = QUICK_EPOCHS
        if output_dir is None:
            output_dir = QUICK_OUTPUT_DIR
        # dossier de test jetable : on repart toujours de zéro
        if Path(output_dir).exists():
            shutil.rmtree(output_dir)
    else:
        if output_dir is None:
            output_dir = OUTPUT_DIR
        # Un entraînement complet coûte ~2 h : on refuse de l'écraser par erreur.
        # Ce test est fait AVANT de charger le modèle, pour échouer tout de suite.
        if list(Path(output_dir).glob("epoch_*")):
            raise SystemExit(f"{output_dir} contient déjà un entraînement : supprime-le ou change de version")

    model, tokenizer = load_model_4bit()
    model = add_lora(model)

    train_conversations = load_split("train")
    val_conversations = load_split("val")
    if quick:
        train_conversations = train_conversations[:QUICK_TRAIN_EXAMPLES]
        val_conversations = val_conversations[:QUICK_VAL_EXAMPLES]
    train_examples = [tokenize_example(messages, tokenizer) for messages in train_conversations]
    val_examples = [tokenize_example(messages, tokenizer) for messages in val_conversations]
    print(f"{len(train_examples)} exemples train, {len(val_examples)} exemples val, {num_epochs} epochs")

    # Un seul optimiseur pour tout l'entraînement : il garde sa mémoire d'une epoch à l'autre.
    # Il ne reçoit que les paramètres entraînables (A et B).
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(output_dir) / "historique_loss.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        csv.writer(file).writerow(["epoch", "loss_train", "loss_val", "secondes"])

    torch.cuda.reset_peak_memory_stats()

    results = []
    for epoch in range(1, num_epochs + 1):
        start_time = time.perf_counter()

        # Seed différente à chaque epoch : train_steps repart de son epoch 1 à chaque appel ;
        # sans seed différente, l'ordre de mélange serait le même à chaque epoch.
        losses, _ = train_steps(model, train_examples, num_epochs=1, optimizer=optimizer, seed=SEED + epoch)
        # loss train = moyenne pendant l'epoch (le modèle s'améliore en cours de route) ;
        # loss val = mesurée à la fin de l'epoch : les deux ne sont pas exactement comparables.
        loss_train = sum(losses) / len(losses)
        loss_val = evaluate_loss(model, val_examples)

        adapter_dir = f"{output_dir}/epoch_{epoch}"
        model.save_pretrained(adapter_dir)
        seconds = time.perf_counter() - start_time

        # On ouvre en mode ajout et on écrit tout de suite : si l'entraînement plante,
        # les epochs terminées restent enregistrées.
        with open(csv_path, "a", encoding="utf-8", newline="") as file:
            csv.writer(file).writerow([epoch, f"{loss_train:.4f}", f"{loss_val:.4f}", f"{seconds:.0f}"])

        print(f"epoch {epoch} : loss train {loss_train:.4f}, loss val {loss_val:.4f}, {seconds:.0f} s")
        results.append(
            {"epoch": epoch, "loss_train": loss_train, "loss_val": loss_val, "seconds": seconds, "adapter_dir": adapter_dir}
        )

    # On garde l'epoch où la loss de validation est minimale et pas forcément la dernière,
    # car après elle le modèle peut commencer à apprendre par cœur (surapprentissage).
    best = results[0]
    for result in results:
        if result["loss_val"] < best["loss_val"]:
            best = result
    print(f"meilleure epoch : {best['epoch']} (loss val {best['loss_val']:.4f}) → adaptateur : {best['adapter_dir']}")

    # Le reste du projet (generate_response, Massi) chargera toujours {output_dir}/best,
    # sans avoir à lire l'affichage pour connaître la meilleure epoch.
    best_dir = f"{output_dir}/best"
    shutil.copytree(best["adapter_dir"], best_dir)
    with open(f"{best_dir}/meilleure_epoch.txt", "w", encoding="utf-8") as file:
        file.write(f"{best['epoch']}\n")
    print(f"meilleur adaptateur copié dans : {best_dir}")

    peak_memory_in_gb = torch.cuda.max_memory_allocated() / 1024**3  # octets -> Go
    print(f"mémoire GPU maximale : {peak_memory_in_gb:.2f} Go")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement complet QLoRA du tuteur.")
    parser.add_argument("--epochs", type=int, default=3, help="nombre d'epochs (ignoré avec --quick, qui en fait 2)")
    parser.add_argument("--quick", action="store_true", help="test rapide : 16 exemples train, 8 val, 2 epochs")
    args = parser.parse_args()
    run_training(args.epochs, args.quick)
