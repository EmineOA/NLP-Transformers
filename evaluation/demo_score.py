"""DÉMONSTRATION : 8 scénarios, prédictions inventées, AUCUN modèle exécuté."""
from pathlib import Path

from score import load_jsonl, parse_response, score_suite

root = Path(__file__).parent
cases = load_jsonl(root / "cases_v0.1.jsonl")
# Prédictions indépendantes fabriquées exprès pour montrer le calcul de scores.
# N2 est intentionnellement MAL classé pour observer la diminution de l'accuracy.
choix_fictifs = {
    ("N1", 1): "PARFAIT",
    ("N2", 1): "INCORRECT",  # erreur volontaire : attendu PARTIEL
    ("N3", 1): "INCORRECT",
    ("N4", 1): "PARTIEL",
    ("N4", 2): "PARFAIT",
}
predictions = []
for case in cases:
    if case["type"] == "parser":
        predictions.append({"id": case["id"], "tour": 1, **parse_response(case["raw_response"])})
        continue
    for n, _ in enumerate(case["turns"], 1):
        key = (case["id"], n)
        if key in choix_fictifs:
            prediction = parse_response(f"[VERDICT: {choix_fictifs[key]}]\nRetour factice.")
        else:
            prediction = {
                "verdict": None,
                "valid_format": True,
                "feedback": "Je ne peux pas traiter cette demande dans ce parcours.",
                "raw_response": "Je ne peux pas traiter cette demande dans ce parcours.",
                # Il s'agit de relectures fictives, pas de vraies évaluations humaines.
            }
        predictions.append({"id": case["id"], "tour": n, **prediction})

details, scores = score_suite(cases, predictions)
print("*** DÉMONSTRATION AVEC PRÉDICTIONS FABRIQUÉES, PAS QUWEN ***")
for r in details:
    attendu = r["attendu"] if r["attendu"] is not None else "refus/format"
    print(f"{r['id']} tour {r['tour']}: attendu={attendu}, obtenu={r['observe']}, format={r['format_valide']}")
print("\nAccuracy des verdicts (fictive) :", scores["accuracy_verdict"])
print("Taux des refus formels (fictif) :", scores["taux_refus_formels"])
print("Refus effectivement corrects :", scores["taux_refus_effectifs_revus"], "(non relus)")
