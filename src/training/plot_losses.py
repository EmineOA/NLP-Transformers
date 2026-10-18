"""Courbes de loss de l'entraînement QLoRA (bloc C).

Ce fichier trace la loss train et la loss val de chaque entraînement, epoch après epoch.
La loss train est l'erreur sur les exemples appris ; la loss val est l'erreur sur des exemples jamais vus.
Quand la loss train continue de baisser alors que la loss val remonte, le modèle apprend par cœur (surapprentissage).
Les historiques lus sont ceux écrits par run_training.py (historique_loss.csv).
"""

# Lancer depuis la racine : python -m src.training.plot_losses

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : on écrit seulement un fichier PNG
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Couleurs de la charte : fond, textes, séries, grille et bordures.
BACKGROUND_COLOR = "#fcfcfb"
TITLE_COLOR = "#0b0b0b"
SECONDARY_COLOR = "#52514e"
TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
GRID_COLOR = "#e6e6e3"
BORDER_COLOR = "#c3c2b7"

MAIN_TITLE = "Loss train et val par epoch (3 epochs ; meilleure epoch = loss val minimale)"


def read_loss_history(csv_path: str) -> dict:
    """Lit un historique_loss.csv et retourne une liste par colonne utile.

    Paramètres :
        csv_path : chemin du fichier (colonnes epoch, loss_train, loss_val, secondes).

    Retour :
        {"epochs": [int], "loss_train": [float], "loss_val": [float]}.

    Exemple :
        read_loss_history("experiments/results/train_v0.2/historique_loss.csv")["epochs"]  # [1, 2, 3]
    """
    history = {"epochs": [], "loss_train": [], "loss_val": []}
    with open(csv_path, encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            history["epochs"].append(int(row["epoch"]))
            history["loss_train"].append(float(row["loss_train"]))
            history["loss_val"].append(float(row["loss_val"]))
    return history


def find_best_epoch(history: dict) -> int:
    """Retourne l'epoch où la loss val est la plus basse.

    Paramètres :
        history : dict retourné par read_loss_history.

    Retour :
        le numéro de la meilleure epoch.

    Exemple :
        find_best_epoch({"epochs": [1, 2], "loss_train": [1.5, 1.0], "loss_val": [1.4, 1.3]})  # 2
    """
    best_index = history["loss_val"].index(min(history["loss_val"]))
    return history["epochs"][best_index]


def draw_one_panel(axis, title: str, history: dict) -> None:
    """Dessine les deux courbes, l'anneau de la meilleure epoch et les étiquettes dans un panneau.

    Paramètres :
        axis : le panneau matplotlib où dessiner.
        title : titre du panneau (le nom du run).
        history : dict retourné par read_loss_history.

    Retour :
        rien (le panneau est modifié sur place).

    Exemple :
        draw_one_panel(axis, "v0.2", read_loss_history("a/historique_loss.csv"))
    """
    epochs = history["epochs"]
    best_epoch = find_best_epoch(history)
    best_loss_val = history["loss_val"][epochs.index(best_epoch)]

    axis.set_facecolor(BACKGROUND_COLOR)
    axis.set_title(title, color=TITLE_COLOR, fontsize=11)
    axis.set_xlabel("epoch", color=SECONDARY_COLOR)
    axis.set_ylabel("loss", color=SECONDARY_COLOR)
    axis.set_xticks(epochs)  # epochs entiers seulement
    # +0.6 à droite : place pour les étiquettes "train" et "val" en bout de ligne
    axis.set_xlim(min(epochs) - 0.4, max(epochs) + 0.6)
    axis.tick_params(colors=SECONDARY_COLOR)

    # set_axisbelow(True) : la grille passe sous les courbes. Grille horizontale seulement, très discrète.
    axis.set_axisbelow(True)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    # spines = les bordures du cadre ; on retire haut et droite pour alléger
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(BORDER_COLOR)

    # Trait plein pour train, tirets pour val : la couleur ne suffit pas
    # (daltonisme, impression en noir et blanc), le style de trait distingue aussi les deux séries.
    axis.plot(epochs, history["loss_train"], color=TRAIN_COLOR, linestyle="-", linewidth=2, marker="o", markersize=6, label="loss train")
    axis.plot(epochs, history["loss_val"], color=VAL_COLOR, linestyle="--", linewidth=2, marker="o", markersize=6, label="loss val")

    # Anneau creux autour du point val de la meilleure epoch : il le signale sans le recouvrir ni ajouter de couleur.
    # zorder=1.5 : au-dessus de la grille, sous les courbes (zorder 2 par défaut).
    axis.plot(
        [best_epoch], [best_loss_val], marker="o", markersize=16, markerfacecolor=BACKGROUND_COLOR,
        markeredgecolor=VAL_COLOR, markeredgewidth=2, linestyle="none", zorder=1.5,
    )
    # "offset points" : décalage en points d'écran, pas en valeurs de loss
    axis.annotate(
        f"meilleure epoch\n(loss val {best_loss_val:.2f})", xy=(best_epoch, best_loss_val), xytext=(0, 26),
        textcoords="offset points", ha="center", color=SECONDARY_COLOR, fontsize=9,
    )

    # Étiquettes directes en bout de ligne : on lit la série sans aller chercher la légende.
    for losses, name in ((history["loss_train"], "train"), (history["loss_val"], "val")):
        axis.annotate(
            name, xy=(epochs[-1], losses[-1]), xytext=(9, 0), textcoords="offset points",
            va="center", color=SECONDARY_COLOR, fontsize=9,
        )


def plot_loss_history(runs: list, output_path: str) -> None:
    """Trace un panneau par entraînement et écrit le graphique en PNG.

    Paramètres :
        runs : liste de (titre, chemin_csv), un panneau par élément.
        output_path : chemin du PNG à écrire.

    Retour :
        rien (le fichier est écrit sur le disque).

    Exemple :
        plot_loss_history([("v0.2", "a/historique_loss.csv")], "courbes.png")
    """
    histories = [read_loss_history(csv_path) for _, csv_path in runs]

    # Petits multiples : un panneau par run, côte à côte, pour comparer les entraînements d'un coup d'œil.
    # Même axe y (sharey) : on compare les deux entraînements à la même échelle ; set_ylim sur un panneau s'applique aux autres.
    # squeeze=False : axes est toujours un tableau 2D, même avec un seul run (sinon axes[0] planterait).
    figure, axes = plt.subplots(1, len(runs), figsize=(10, 4), sharey=True, squeeze=False)
    axes = axes[0]
    figure.patch.set_facecolor(BACKGROUND_COLOR)
    figure.suptitle(MAIN_TITLE, color=TITLE_COLOR, fontsize=12)

    for axis, (title, _), history in zip(axes, runs, histories):
        draw_one_panel(axis, title, history)

    # marge en haut pour l'annotation, en bas pour la légende
    all_losses = [loss for history in histories for loss in history["loss_train"] + history["loss_val"]]
    axes[0].set_ylim(min(all_losses) - 0.15, max(all_losses) + 0.25)

    # Une seule légende, dans le 1er panneau : les couleurs sont les mêmes dans les deux.
    # L'anneau n'a pas de label dans draw_one_panel : on crée son entrée à la main.
    ring_handle = Line2D(
        [], [], marker="o", markersize=12, markerfacecolor=BACKGROUND_COLOR, markeredgecolor=VAL_COLOR,
        markeredgewidth=2, linestyle="none",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    legend = axes[0].legend(handles + [ring_handle], labels + ["meilleure epoch (loss val minimale)"], loc="lower left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(SECONDARY_COLOR)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, facecolor=BACKGROUND_COLOR)
    plt.close(figure)


if __name__ == "__main__":
    # Test sans GPU de find_best_epoch sur un cas fabriqué à la main : le minimum est à l'epoch 2.
    fake_history = {"epochs": [1, 2, 3], "loss_train": [1.5, 1.0, 0.7], "loss_val": [1.4, 1.2, 1.3]}
    assert find_best_epoch(fake_history) == 2

    runs = [
        ("v0.2 — RTX 4070, lr 2e-4", "experiments/results/train_v0.2/historique_loss.csv"),
        ("v0.3 — Colab A100, lr 1e-4", "experiments/results/train_v0.3/historique_loss.csv"),
    ]
    output_path = "experiments/results/courbes_loss.png"
    plot_loss_history(runs, output_path)

    # 10 Ko : un PNG vide ou corrompu fait quelques Ko, un vrai graphique en dpi 200 est bien plus gros
    assert Path(output_path).stat().st_size > 10_000
    expected_best_epochs = [1, 2]
    for (_, csv_path), expected_best_epoch in zip(runs, expected_best_epochs):
        history = read_loss_history(csv_path)
        assert len(history["epochs"]) == 3
        assert find_best_epoch(history) == expected_best_epoch
    print(f"courbes écrites dans : {output_path}")
