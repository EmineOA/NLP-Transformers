"""Contrôles automatiques des lignes générées (étape 3 du README), sans LLM.

Lit par défaut le CSV fusionné dataset_source.csv (les JSON bruts sont optionnels
avec --dossier), puis écrit generation/rapport_validation.csv : une ligne par entrée,
avec son statut
(OK, ALERTE ou REJET) et les raisons. Les fichiers générés ne sont pas modifiés.

- REJET  : la ligne ne respecte pas le format ou les règles du feedback -> à regénérer.
- ALERTE : la ligne est valide mais suspecte (tic d'IA, doublon...) -> à relire.

Usage : python data/dialogue/classification/valider_dataset.py [--question QUESTION_ID]
"""
import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path
import subprocess
from collections import Counter, defaultdict

ICI = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(os.path.dirname(os.path.dirname(ICI)))
SQUELETTE = os.path.join(ICI, "dataset_source.csv")
GENERATION = os.path.join(ICI, "generation")
FICHES = os.path.join(RACINE, "data", "concepts", "json")
RAPPORT = os.path.join(GENERATION, "rapport_validation.csv")

CHAMPS = {"id", "historique", "reponse_utilisateur", "feedback"}
MARQUEUR = {"PARFAIT": "[VERDICT: PARFAIT]", "PARTIEL": "[VERDICT: PARTIEL]",
            "INCORRECT": "[VERDICT: INCORRECT]", "HORS_PERIMETRE": "[HORS_PERIMETRE]"}
# Tics listés dans prompt_a_donner.md (sections 5.2 et 5.4)
TICS = [r"en d[ée]coule", r"\bclaire?s?\b", r"\bclairement\b", r"montre bien", r"\bsolide",
        r"\bexactement\b", r"n'h[ée]site", r"il est important", r"\bcrucia", r"\bessentiel",
        r"\bpertinent", r"\bremarquable", r"\bd[ée]montre", r"met en lumi[èe]re",
        r"tout est l[àa]", r"c'est la cl[ée]", r"\bau fond\b", r"honn[êe]tement ?\?",
        r"alors,? voyons", r"excellente question", r"j'esp[èe]re que"]
REFORMULATION = re.compile(r"^(ta|ton|tes) (r[èe]gle|r[ée]ponse|raisonnement|explication|argument)", re.I)
SEUIL_DOUBLON = 0.6


def mots(texte):
    return set(re.findall(r"\w+", texte.lower()))


def jaccard(a, b):
    a, b = mots(a), mots(b)
    return len(a & b) / len(a | b) if a and b else 0.0


def fin_question(texte):
    return texte.rstrip(" »\"')").endswith("?")


def textes_externes():
    """Cas de test du bloc D et cas de développement du bloc B (lus via git si disponibles)."""
    textes = []
    for ref in ("origin/evaluation:evaluation/cases_v0.1.jsonl", "origin/main:experiments/dev_cases_bloc_b.jsonl"):
        try:
            sortie = subprocess.run(["git", "-C", RACINE, "show", ref], capture_output=True,
                                    text=True, encoding="utf-8", check=True).stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        for ligne in filter(None, sortie.splitlines()):
            cas = json.loads(ligne)
            textes += [t["user_input"] for t in cas.get("turns", [])] + [cas.get("user_input", "")]
    return [t for t in textes if t]


def exemples_fiches():
    exemples = defaultdict(list)
    for chemin in glob.glob(os.path.join(FICHES, "*.json")):
        for q in json.loads(Path(chemin).read_text(encoding="utf-8"))["questions"]:
            exemples[q["id"]] += [n.get("exemple", "") for n in q["niveaux"].values() if n.get("exemple")]
    return exemples


def verifier_historique(ligne, historique):
    attendus = [e for e in ligne["labels_historique"].split(" | ") if e]
    elements = historique.split("\n") if historique else []
    if len(elements) != len(attendus):
        return [f"historique : {len(elements)} éléments au lieu de {len(attendus)}"]
    erreurs = []
    for n, (element, attendu) in enumerate(zip(elements, attendus), 1):
        cible, label = attendu.rsplit(":", 1)
        if " essai " in cible:
            qid, essai = cible.split(" essai ")
            motif = re.escape(f"[{qid} – essai {essai}] ") + r".+ → " + re.escape(MARQUEUR[label])
        else:
            motif = re.escape(f"[{cible}] ") + r".+"
        if not re.fullmatch(motif, element.strip()):
            erreurs.append(f"historique élément {n} : attendu « {attendu} »")
    return erreurs


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--question", help="ne valider qu'une question_id")
    cli.add_argument("--dossier", help="vérifier d'anciennes sorties JSON plutôt que le CSV fusionné")
    args = cli.parse_args()
    global RAPPORT
    report_folder = args.dossier or GENERATION
    RAPPORT = os.path.join(report_folder, "rapport_validation.csv")

    with open(SQUELETTE, encoding="utf-8-sig", newline="") as f:
        source_rows = list(csv.DictReader(f, delimiter=";"))
    squelette = {line["id"]: line for line in source_rows}
    exemples, externes = exemples_fiches(), textes_externes()

    entrees = []  # (generateur, fichier, entree)
    if args.dossier:
        for chemin in sorted(glob.glob(os.path.join(args.dossier, "*", "*.json"))):
            generateur, nom = os.path.basename(os.path.dirname(chemin)), os.path.basename(chemin)[:-5]
            if args.question and nom != args.question:
                continue
            for entree in json.loads(Path(chemin).read_text(encoding="utf-8")):
                entrees.append((generateur, nom, entree))
    else:
        for line in source_rows:
            if not args.question or line["question_id"] == args.question:
                entrees.append((line["generateur"], line["question_id"], {key: line[key] for key in CHAMPS}))
    if not entrees:
        raise SystemExit("Aucune ligne à contrôler : vérifier le CSV, le dossier ou la question.")

    resultats, vus = [], Counter(e.get("id") for _, _, e in entrees)
    debuts = Counter((nom, " ".join(e.get("feedback", "").split()[:3]).lower()) for _, nom, e in entrees)
    par_question = defaultdict(list)
    for g, nom, e in entrees:
        par_question[nom].append(e)

    for generateur, nom, e in entrees:
        rejets, alertes = [], []
        ligne = squelette.get(e.get("id"))
        if set(e) != CHAMPS:
            rejets.append(f"champs : {sorted(set(e) ^ CHAMPS)}")
        if ligne is None:
            rejets.append("id absent du squelette")
            resultats.append((e.get("id"), generateur, nom, "?", "REJET", rejets))
            continue
        if ligne["question_id"] != nom or ligne["generateur"] != generateur:
            rejets.append("ligne rangée dans le mauvais fichier ou attribuée à un autre générateur")
        if vus[e["id"]] > 1:
            rejets.append("id en double")
        reponse, feedback, historique = (str(e.get(k, "")) for k in ("reponse_utilisateur", "feedback", "historique"))
        label = ligne["label"]

        # 1. Structure
        if not reponse.strip() or not feedback.strip():
            rejets.append("réponse ou feedback vide")
        if ";" in reponse + feedback + historique:
            rejets.append("caractère ;")
        if "\n" in reponse or "\n" in feedback:
            rejets.append("saut de ligne dans la réponse ou le feedback")
        if re.search(r"\[VERDICT|\[HORS_PERIMETRE", reponse + feedback, re.I):
            rejets.append("marqueur de verdict dans la réponse ou le feedback")
        # 2. Historique
        rejets += verifier_historique(ligne, historique)
        # 3. Règles du feedback
        if label == "PARFAIT" and "?" in feedback:
            rejets.append("PARFAIT avec une question")
        if label in ("PARTIEL", "INCORRECT") and not fin_question(feedback):
            rejets.append(f"{label} sans question finale")
        if label == "HORS_PERIMETRE" and "?" not in feedback:
            rejets.append("HORS_PERIMETRE sans relance")

        # 5. Tics et 6. reformulation
        tics = [t for t in TICS if re.search(t, feedback, re.I)]
        if tics:
            alertes.append("tics : " + ", ".join(tics))
        if REFORMULATION.match(feedback.strip()):
            alertes.append("feedback qui commence par reformuler")
        # 7. Diversité
        if debuts[(nom, " ".join(feedback.split()[:3]).lower())] > 2:
            alertes.append("début de feedback répété dans la question")
        for autre in par_question[nom]:
            if autre is not e and jaccard(reponse, autre.get("reponse_utilisateur", "")) > SEUIL_DOUBLON:
                alertes.append(f"réponse proche de {autre['id']}")
        if any(jaccard(reponse, x) > SEUIL_DOUBLON for x in exemples[nom]):
            alertes.append("réponse proche d'un exemple de la fiche")
        if ligne["style"] == "court" and len(re.findall(r"[.!?]+(\s|$)", reponse)) > 3:
            alertes.append("style court mais réponse longue")
        # 8. Séparation avec l'évaluation
        if any(jaccard(reponse, x) > SEUIL_DOUBLON for x in externes):
            alertes.append("réponse proche d'un cas de test ou de développement")

        statut = "REJET" if rejets else "ALERTE" if alertes else "OK"
        resultats.append((e["id"], generateur, nom, label, statut, rejets + alertes))

    os.makedirs(report_folder, exist_ok=True)
    with open(RAPPORT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "generateur", "question_id", "label", "statut", "raisons"])
        for r in resultats:
            w.writerow([*r[:5], " | ".join(r[5]).replace(";", ",")])

    # Résumé
    print(f"{len(resultats)} lignes contrôlées, {len(externes)} cas externes comparés")
    print("statuts :", dict(Counter(r[4] for r in resultats)))
    taux = Counter((r[1], r[2], r[4]) for r in resultats)
    for g, q in sorted({(r[1], r[2]) for r in resultats}):
        n = sum(taux[(g, q, s)] for s in ("OK", "ALERTE", "REJET"))
        attendu = sum(1 for l in squelette.values() if l["question_id"] == q and l["generateur"] == g)
        print(f"  {g:16} {q:36} {n:3}/{attendu:<3} OK {taux[(g, q, 'OK')]:3}  ALERTE {taux[(g, q, 'ALERTE')]:3}  REJET {taux[(g, q, 'REJET')]:3}")
    for r in resultats:
        if r[4] == "REJET":
            print("REJET", r[0], ":", " | ".join(r[5]))
    print("Rapport :", RAPPORT)


if __name__ == "__main__":
    main()
