"""Attribue chaque ligne du squelette à un modèle générateur, par tirage aléatoire.

Méthode : tirage sans remise. Tant qu'il reste des places pour Opus, on tire un
indice au hasard parmi les x lignes restantes, puis parmi les x - 1 suivantes, etc.
Les lignes tirées vont à Opus, les autres à Sol. La graine est fixe : relancer le
script redonne exactement la même attribution.

Usage : python data/dialogue/classification/attribuer_generateurs.py
"""
import csv
import os
import random
from collections import Counter

ICI = os.path.dirname(os.path.abspath(__file__))
FICHIER = os.path.join(ICI, "dataset_source.csv")

GRAINE = 2026
GENERATEURS = {"claude-opus-5.5": 472, "sol-6.1": 531}


def main():
    with open(FICHIER, encoding="utf-8-sig", newline="") as f:
        lignes = list(csv.DictReader(f, delimiter=";"))
    colonnes = [c for c in lignes[0].keys() if c != "generateur"] + ["generateur"]
    if len(lignes) != sum(GENERATEURS.values()):
        raise ValueError(f"{len(lignes)} lignes, mais {sum(GENERATEURS.values())} places prévues")

    rng = random.Random(GRAINE)
    restantes = list(range(len(lignes)))
    premier, second = GENERATEURS
    tirees = set()
    for _ in range(GENERATEURS[premier]):
        # indice aléatoire entre 0 et x - 1, x = nombre de lignes encore libres
        k = rng.randrange(len(restantes))
        tirees.add(restantes.pop(k))
    for i, ligne in enumerate(lignes):
        ligne["generateur"] = premier if i in tirees else second

    with open(FICHIER, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=colonnes, delimiter=";")
        w.writeheader()
        w.writerows(lignes)

    # Contrôles
    print("générateurs :", dict(Counter(l["generateur"] for l in lignes)))
    for cle in ("label", "split", "essai_question"):
        print(f"{cle:15}", dict(Counter((l[cle], l["generateur"]) for l in lignes)))
    par_q = Counter((l["question_id"], l["generateur"]) for l in lignes)
    nb = sorted(par_q[(q, premier)] for q in {l["question_id"] for l in lignes})
    print(f"lignes {premier} par question : min {nb[0]}, max {nb[-1]}")


if __name__ == "__main__":
    main()
