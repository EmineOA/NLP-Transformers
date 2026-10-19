"""Génère le squelette du dataset d'entraînement (data/dialogue/classification/dataset_source.csv).

Une ligne = une réponse de l'élève à évaluer. Le squelette fixe tout ce que le LLM
générateur doit respecter ; le LLM remplit ensuite `historique`, `reponse_utilisateur`
et `feedback`.

Colonnes :
- essai_question     : numéro de l'essai sur la question actuelle (1, 2 ou 3)
- labels_historique  : niveau imposé de chaque élément de l'historique
    * questions précédentes du concept -> label du DERNIER essai (PARFAIT/PARTIEL/INCORRECT)
    * essais précédents sur la question actuelle -> label de chaque essai (jamais PARFAIT,
      sinon la question serait terminée)
- historique         : vide (rempli par le LLM)
- cas / style        : critère de la fiche à illustrer, style d'écriture

Usage : python data/dialogue/classification/generer_squelette.py
"""
import csv
import glob
import json
import os
from collections import Counter

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(os.path.dirname(ICI))
DOSSIER_JSON = os.path.join(RACINE, "concepts", "json")
SORTIE = os.path.join(ICI, "dataset_source.csv")

# Refus : convention du moteur (src/agents/baseline.py) -> marqueur [HORS_PERIMETRE].
# Dans les fiches, le niveau correspondant s'appelle toujours "refus".
CLE_NIVEAU = {"PARFAIT": "parfait", "PARTIEL": "partiel", "INCORRECT": "incorrect", "HORS_PERIMETRE": "refus"}

# Lignes par question et par label (17 questions x 59 = 1003 lignes)
NB_PAR_LABEL = {"PARFAIT": 16, "PARTIEL": 16, "INCORRECT": 16, "HORS_PERIMETRE": 11}

# Répartition des essais : 16 lignes -> 8 essai 1, 4 essai 2, 4 essai 3
MOTIF_ESSAIS = {
    "PARFAIT": [1, 2, 1, 3],
    "PARTIEL": [1, 3, 1, 2],
    "INCORRECT": [1, 2, 1, 3],
    "HORS_PERIMETRE": [1, 2, 3, 1, 3, 2, 1, 2, 3, 1, 1],  # 5 / 3 / 3
}

# Niveau imposé du dernier essai des questions précédentes (60 % PARFAIT)
CYCLE_QUESTIONS_PRECEDENTES = ["PARFAIT", "PARTIEL", "PARFAIT", "INCORRECT", "PARFAIT"]
# Niveau imposé des essais ratés précédents sur la question actuelle
CYCLE_ESSAIS_PRECEDENTS = ["INCORRECT", "PARTIEL", "HORS_PERIMETRE", "PARTIEL", "INCORRECT"]

STYLES = ["court", "hésitant", "fautes d'orthographe"]

# Split groupé par question (toutes les lignes d'une question dans le même split) :
# 11 questions train, 3 val, 3 test.
QUESTIONS_VAL = {"dilemme-du-tramway_q2", "monty-hall_q3", "plier-une-feuille-42-fois_q2"}
QUESTIONS_TEST = {"bateau-de-thesee_q3", "doute-de-descartes_q1", "infinis-plus-grands-que-l-infini_q2"}

COLONNES = [
    "id", "concept_id", "question_id", "enonce", "essai_question",
    "labels_historique", "historique", "reponse_utilisateur",
    "label", "feedback", "cas", "style", "split",
]


def criteres(question, label):
    if label == "PARFAIT":
        return ["tous les critères"]
    crits = question["niveaux"][CLE_NIVEAU[label]]["criteres"]
    if label == "HORS_PERIMETRE":
        crits = [c for c in crits if "hors-sujet" not in c.lower()]
        crits = crits + ["autre demande hors cadre (au choix)"]
    return crits


def main():
    lignes = []
    for chemin in sorted(glob.glob(os.path.join(DOSSIER_JSON, "*.json"))):
        concept = json.load(open(chemin, encoding="utf-8"))
        questions = concept["questions"]
        for iq, q in enumerate(questions):
            precedentes = [p["id"] for p in questions[:iq]]
            compteur_prec = iq          # décale les cycles d'une question à l'autre
            compteur_essai = iq * 2
            for label, n in NB_PAR_LABEL.items():
                crits = criteres(q, label)
                motif = MOTIF_ESSAIS[label]
                for i in range(n):
                    essai = motif[i % len(motif)]
                    elements = []
                    for k, pid in enumerate(precedentes):
                        lab = CYCLE_QUESTIONS_PRECEDENTES[(compteur_prec + k * 2) % len(CYCLE_QUESTIONS_PRECEDENTES)]
                        elements.append(f"{pid}:{lab}")
                    compteur_prec += 1
                    for e in range(1, essai):
                        lab = CYCLE_ESSAIS_PRECEDENTS[compteur_essai % len(CYCLE_ESSAIS_PRECEDENTS)]
                        compteur_essai += 1
                        elements.append(f"{q['id']} essai {e}:{lab}")
                    lignes.append({
                        "id": f"{q['id']}_{label.lower()}_{i + 1:02d}",
                        "concept_id": concept["id"],
                        "question_id": q["id"],
                        "enonce": q["enonce"],
                        "essai_question": essai,
                        "labels_historique": " | ".join(elements),
                        "historique": "",
                        "reponse_utilisateur": "",
                        "label": label,
                        "feedback": "",
                        "cas": crits[i % len(crits)],
                        "style": STYLES[(i + iq) % len(STYLES)],
                        "split": ("test" if q["id"] in QUESTIONS_TEST
                                  else "val" if q["id"] in QUESTIONS_VAL else "train"),
                    })

    with open(SORTIE, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLONNES, delimiter=";")
        w.writeheader()
        w.writerows(lignes)

    # Contrôles
    print("lignes :", len(lignes))
    print("labels :", dict(Counter(l["label"] for l in lignes)))
    print("essais :", dict(Counter(l["essai_question"] for l in lignes)))
    print("split  :", dict(Counter(l["split"] for l in lignes)))
    print("historique vide :", sum(1 for l in lignes if not l["labels_historique"]))
    niv = Counter()
    for l in lignes:
        for el in filter(None, l["labels_historique"].split(" | ")):
            niv[("essai" if " essai " in el else "question", el.rsplit(":", 1)[1])] += 1
    print("niveaux dans l'historique :", dict(niv))


if __name__ == "__main__":
    main()
