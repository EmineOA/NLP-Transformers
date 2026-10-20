"""Courbes de loss de l'entraînement QLoRA (bloc C).

Ce fichier trace la loss train et la loss val de chaque entraînement, epoch après epoch.
La loss train est l'erreur sur les exemples appris ; la loss val est l'erreur sur des exemples jamais vus.
Quand la loss train continue de baisser alors que la loss val remonte, le modèle apprend par cœur (surapprentissage).
Les historiques lus sont ceux écrits par run_training.py :
historique_loss.csv (une ligne par epoch) pour les courbes par epoch,
historique_pas.csv et historique_val.csv (une ligne par pas, une par mesure val) pour la courbe fine.
"""

# Lancer depuis la racine : python -m src.training.plot_losses

import csv
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : on écrit seulement un fichier PNG
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from src.training.train import ACCUMULATION_STEPS

# Couleurs de la charte : fond, textes, séries, grille et bordures.
BACKGROUND_COLOR = "#fcfcfb"
TITLE_COLOR = "#0b0b0b"
SECONDARY_COLOR = "#52514e"
TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
GRID_COLOR = "#e6e6e3"
BORDER_COLOR = "#c3c2b7"

MAIN_TITLE = "Loss train et val par epoch (meilleure epoch = loss val minimale)"


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


def read_csv_columns(csv_path) -> dict:
    """Lit un fichier CSV et retourne chaque colonne sous forme de liste de nombres.

    Paramètres :
        csv_path : chemin d'un CSV dont toutes les colonnes sont numériques (historique_pas.csv, historique_val.csv).

    Retour :
        {nom_de_colonne: [float, ...]}.

    Exemple :
        read_csv_columns("adapters/v0.2/historique_val.csv")["loss_val"]  # [1.79, 1.72]
    """
    columns = {}
    with open(csv_path, encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            for name, value in row.items():
                columns.setdefault(name, []).append(float(value))
    return columns


def moving_average(values: list, window: int = 10) -> list:
    """Lisse une série : chaque point devient la moyenne de lui-même et des window - 1 points précédents.

    Paramètres :
        values : la série (une loss par pas).
        window : nombre de points moyennés.

    Retour :
        une liste de la même longueur ; au début, la moyenne porte sur les points disponibles.

    Exemple :
        moving_average([1, 3, 5], window=2)  # [1.0, 2.0, 4.0]
    """
    smoothed = []
    for index in range(len(values)):
        recent = values[max(0, index - window + 1):index + 1]
        smoothed.append(sum(recent) / len(recent))
    return smoothed


def find_epoch_starts(steps: list, epochs: list) -> list:
    """Retourne, pour chaque epoch à partir de la 2e, le pas où elle commence.

    Paramètres :
        steps : numéros de pas (colonne pas de historique_pas.csv).
        epochs : epoch de chaque pas (même longueur).

    Retour :
        liste de (premier_pas_de_l_epoch, numéro_d_epoch).

    Exemple :
        find_epoch_starts([1, 2, 3, 4], [1, 1, 2, 2])  # [(3, 2)]
    """
    starts = []
    for index in range(1, len(steps)):
        if epochs[index] != epochs[index - 1]:
            starts.append((steps[index], int(epochs[index])))
    return starts


def draw_run_curves(axis, steps: list, loss_train: list, val_steps: list, loss_val: list) -> None:
    """Dessine la loss train (brute et lissée) et la loss val d'un entraînement.

    Paramètres :
        axis : le panneau matplotlib où dessiner.
        steps, loss_train : une loss train par pas d'optimisation.
        val_steps, loss_val : une loss val par mesure.

    Retour :
        rien (le panneau est modifié sur place).

    Exemple :
        draw_run_curves(axis, [1, 2], [1.5, 1.4], [2], [1.6])
    """
    # La loss brute varie beaucoup d'un groupe à l'autre : trait fin et clair, pour montrer le bruit sans masquer la tendance.
    axis.plot(steps, loss_train, color=TRAIN_COLOR, alpha=0.35, linewidth=1)
    axis.plot(steps, moving_average(loss_train, 10), color=TRAIN_COLOR, linewidth=2, label="loss train (moyenne glissante sur 10 pas)")
    # Tirets et marqueurs pour val : la couleur ne suffit pas (daltonisme, noir et blanc).
    axis.plot(val_steps, loss_val, color=VAL_COLOR, linestyle="--", linewidth=2, marker="o", markersize=5, label="loss val (mesure sur tout val)")


def draw_run_markers(axis, val_steps: list, loss_val: list, epoch_starts: list) -> None:
    """Marque le meilleur point val et les changements d'epoch.

    Paramètres :
        axis : le panneau matplotlib où dessiner.
        val_steps, loss_val : une loss val par mesure ; le minimum est le meilleur adaptateur.
        epoch_starts : sortie de find_epoch_starts.

    Retour :
        rien (le panneau est modifié sur place).

    Exemple :
        draw_run_markers(axis, [2, 4], [1.8, 1.7], [(3, 2)])
    """
    best_index = loss_val.index(min(loss_val))
    best_step, best_loss = val_steps[best_index], loss_val[best_index]
    axis.axvline(best_step, color=SECONDARY_COLOR, linestyle="--", linewidth=0.8, alpha=0.6)
    # Anneau creux autour du meilleur point val : il le signale sans le recouvrir.
    axis.plot(
        [best_step], [best_loss], marker="o", markersize=14, markerfacecolor="none",
        markeredgecolor=VAL_COLOR, markeredgewidth=2, linestyle="none", label="meilleur adaptateur (loss val minimale)",
    )
    # Le texte est posé en bas de la ligne verticale (y en fraction de la hauteur) : il ne dépend pas des valeurs de loss.
    # Si le meilleur pas est dans la moitié droite, le texte s'écrit vers la gauche pour ne pas sortir du cadre.
    on_right_half = best_step > max(val_steps) / 2
    axis.annotate(
        f"meilleur adaptateur (pas {best_step:.0f}, loss val {best_loss:.2f})", xy=(best_step, 0.02),
        xycoords=axis.get_xaxis_transform(), xytext=(-5 if on_right_half else 5, 0), textcoords="offset points",
        color=SECONDARY_COLOR, fontsize=9, ha="right" if on_right_half else "left", va="bottom",
    )
    # Lignes d'epoch très discrètes, placées juste avant le 1er pas de l'epoch.
    for first_step, epoch in epoch_starts:
        axis.axvline(first_step - 0.5, color=BORDER_COLOR, linewidth=0.8, alpha=0.7)
        axis.text(
            first_step - 0.5, 0.98, f" epoch {epoch}", transform=axis.get_xaxis_transform(),
            color=SECONDARY_COLOR, fontsize=8, ha="left", va="top",
        )


def read_effective_batch_size(output_dir) -> int:
    """Lit le batch effectif (exemples par pas d'optimisation) d'un entraînement.

    Paramètres :
        output_dir : dossier de l'entraînement (avec ou sans config.json).

    Retour :
        batch_size_effectif de config.json s'il existe ; sinon ACCUMULATION_STEPS
        (les anciens runs, comme v0.3, ont un lot de 1 exemple et 8 accumulations).

    Exemple :
        read_effective_batch_size("adapters/v0.4")  # 8
    """
    config_path = Path(output_dir) / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as file:
            config = json.load(file)
        # config.json d'un ancien run : pas de clé batch_size_effectif, on retombe sur la valeur par défaut
        return config.get("batch_size_effectif", ACCUMULATION_STEPS)
    return ACCUMULATION_STEPS


def plot_training_run(output_dir: str, output_path: str, title: str) -> None:
    """Trace la courbe fine d'UN entraînement : loss train par pas, loss val par mesure, meilleur adaptateur.

    Paramètres :
        output_dir : dossier de l'entraînement (contient historique_pas.csv et historique_val.csv).
        output_path : chemin du PNG à écrire.
        title : titre du graphique (prudent : décrire, ne pas conclure).

    Retour :
        rien (le fichier est écrit sur le disque).

    Exemple :
        plot_training_run("adapters/v0.4", "courbe_fine.png", "Loss par pas d'optimisation")
    """
    train = read_csv_columns(Path(output_dir) / "historique_pas.csv")
    val = read_csv_columns(Path(output_dir) / "historique_val.csv")

    figure, axis = plt.subplots(figsize=(10, 5))
    figure.patch.set_facecolor(BACKGROUND_COLOR)
    axis.set_facecolor(BACKGROUND_COLOR)
    axis.set_title(title, color=TITLE_COLOR, fontsize=12)
    axis.set_xlabel(f"pas d'optimisation (1 pas = {read_effective_batch_size(output_dir)} exemples)", color=SECONDARY_COLOR)
    axis.set_ylabel("loss", color=SECONDARY_COLOR)
    axis.tick_params(colors=SECONDARY_COLOR)
    # un pas est un nombre entier : pas de graduation 1.5
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_axisbelow(True)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(BORDER_COLOR)

    draw_run_curves(axis, train["pas"], train["loss_train"], val["pas"], val["loss_val"])
    draw_run_markers(axis, val["pas"], val["loss_val"], find_epoch_starts(train["pas"], train["epoch"]))

    # Légende sous le graphique : à l'intérieur, elle risquerait de cacher des points.
    legend = axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(SECONDARY_COLOR)
    # bbox_inches="tight" : agrandit l'image pour inclure la légende placée hors du cadre
    figure.savefig(output_path, dpi=200, facecolor=BACKGROUND_COLOR, bbox_inches="tight")
    plt.close(figure)


def format_run_row(run: dict) -> str:
    """Construit la ligne Markdown d'un entraînement à partir de son historique_loss.csv et de son config.json.

    Paramètres :
        run : {"version", "output_dir"} et, pour un dossier sans config.json (v0.2), "gpu" et "learning_rate".

    Retour :
        une ligne de tableau Markdown : version, GPU, taux d'apprentissage, meilleure epoch,
        loss val minimale, loss val par epoch, durée par epoch.

    Exemple :
        format_run_row({"version": "v0.3", "output_dir": "experiments/results/train_v0.3"})
    """
    folder = Path(run["output_dir"])
    history = read_loss_history(folder / "historique_loss.csv")
    # Les durées ne sont pas dans read_loss_history (qui sert aux courbes) : on les relit ici.
    seconds = read_csv_columns(folder / "historique_loss.csv")["secondes"]

    # config.json prime sur les infos passées à la main : c'est ce que le run a vraiment utilisé.
    info = dict(run)
    config_path = folder / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as file:
            config = json.load(file)
        info["gpu"] = config["gpu"]
        info["learning_rate"] = config["learning_rate"]

    learning_rate = f"{info['learning_rate']:.0e}".replace("e-0", "e-")  # 0.0002 -> 2e-4
    durations = " / ".join(f"{value / 60:.0f}" for value in seconds)
    val_losses = " / ".join(f"{value:.4f}" for value in history["loss_val"])
    return (
        f"| {run['version']} | {info['gpu']} | {learning_rate} | {find_best_epoch(history)} "
        f"| {min(history['loss_val']):.4f} | {val_losses} | {durations} |"
    )


def write_runs_table(runs: list, output_path: str) -> None:
    """Écrit un tableau Markdown qui compare les entraînements, un par ligne.

    Paramètres :
        runs : liste de dicts {"version", "output_dir"} (+ "gpu" et "learning_rate" si pas de config.json).
        output_path : chemin du fichier .md à écrire.

    Retour :
        rien (le fichier est écrit sur le disque).

    Exemple :
        write_runs_table([{"version": "v0.3", "output_dir": "experiments/results/train_v0.3"}], "comparaison_runs.md")
    """
    lines = [
        "| Version | GPU | Taux d'apprentissage | Meilleure epoch | Loss val minimale | Loss val par epoch | Durée par epoch (min) |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [format_run_row(run) for run in runs]
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    # Tableau de comparaison : v0.2 n'a pas de config.json, ses infos sont données à la main.
    table_runs = [
        {"version": "v0.2", "output_dir": "experiments/results/train_v0.2", "gpu": "RTX 4070", "learning_rate": 2e-4},
        {"version": "v0.3", "output_dir": "experiments/results/train_v0.3"},
    ]
    table_path = "experiments/results/comparaison_runs.md"
    write_runs_table(table_runs, table_path)
    assert len(Path(table_path).read_text(encoding="utf-8").splitlines()) == 2 + len(table_runs)
    print(f"tableau écrit dans : {table_path}")

    # Petites fonctions de la courbe fine
    assert moving_average([1, 3, 5], window=2) == [1.0, 2.0, 4.0]
    assert find_epoch_starts([1, 2, 3, 4], [1, 1, 2, 2]) == [(3, 2)]

    # Batch effectif : v0.3 n'a pas la clé (repli sur ACCUMULATION_STEPS), un config.json récent la donne.
    assert read_effective_batch_size("experiments/results/train_v0.3") == ACCUMULATION_STEPS
    with tempfile.TemporaryDirectory() as folder:
        Path(folder, "config.json").write_text('{"batch_size_effectif": 16}', encoding="utf-8")
        assert read_effective_batch_size(folder) == 16

    # Courbe fine : seulement si un entraînement a laissé ses fichiers (par ex. adapters/quick après le test rapide).
    # Le PNG de test est supprimé ensuite : il ne sert qu'à vérifier que la fonction tourne.
    run_folders = sorted(Path("adapters").glob("*/historique_pas.csv"))
    if run_folders:
        test_path = "experiments/results/test_courbe_fine.png"
        plot_training_run(str(run_folders[0].parent), test_path, "Test : loss par pas d'optimisation")
        assert Path(test_path).stat().st_size > 10_000
        print(f"courbe fine testée sur : {run_folders[0].parent}")
        Path(test_path).unlink()
