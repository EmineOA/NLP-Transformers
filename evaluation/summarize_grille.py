"""Tableau synthétique de la grille humaine : moyenne par critère et par modèle."""
import csv
import sys

CRITERES = [
    "pertinence_0_2",
    "role_format_0_2",
    "historique_0_2",
    "hors_perimetre_0_2",
    "absence_inventions_0_2",
]


def summarize_grid(rows: list) -> dict:
    """Calcule la moyenne (0 à 2) de chaque critère pour chaque modèle.

    Paramètres :
        rows : lignes du CSV de la grille humaine (liste de dictionnaires).

    Retour :
        {modele: {critere: (moyenne ou None, nombre de notes)}}.

    Exemple :
        summarize_grid([{"modele": "lora", "pertinence_0_2": "2"}])["lora"]["pertinence_0_2"]  # (2.0, 1)
    """
    result = {}
    for row in rows:
        stats = result.setdefault(row["modele"], {c: [] for c in CRITERES})
        for critere in CRITERES:
            value = (row.get(critere) or "").strip()
            # "N/A" = critère non applicable au test : exclu de la moyenne
            if value in ("0", "1", "2"):
                stats[critere].append(int(value))
    return {
        modele: {c: (sum(v) / len(v) if v else None, len(v)) for c, v in crit.items()}
        for modele, crit in result.items()
    }


def _self_test() -> None:
    rows = [
        {"modele": "a", "pertinence_0_2": "2", "historique_0_2": "N/A"},
        {"modele": "a", "pertinence_0_2": "1", "historique_0_2": ""},
    ]
    out = summarize_grid(rows)["a"]
    assert out["pertinence_0_2"] == (1.5, 2)
    assert out["historique_0_2"] == (None, 0)


if __name__ == "__main__":
    _self_test()
    path = sys.argv[1] if len(sys.argv) > 1 else "evaluation/final_v1.0/results/grille_humaine_comparative.csv"
    with open(path, encoding="utf-8-sig") as f:
        summary = summarize_grid(list(csv.DictReader(f)))
    print("| Critère | " + " | ".join(summary) + " |")
    print("|---|" + "---|" * len(summary))
    for critere in CRITERES:
        cells = []
        for modele in summary:
            mean, n = summary[modele][critere]
            cells.append("n/a" if mean is None else f"{mean:.2f} (n={n})")
        print(f"| {critere} | " + " | ".join(cells) + " |")
