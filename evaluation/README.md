# T6.1 — Première évaluation de Brainmaxing

Cet outil s'appuie sur `backlog.md` (E6/T6.1) et les fiches de concept du GitHub fourni. Il ne modifie pas la baseline.

## Fichiers
- `cases_v0.1.jsonl` : **8 scénarios** N1–N5 et X1–X3. N4 a deux tours : **9 observations au total**. Tous les attendus devront être relus avec le bloc A.
- `score.py` : parseur pour les verdicts et scoreur des prédictions structurées du moteur.
- `test_score.py` : tests de logique *avec prédictions fabriquées*, et NON mesure de Qwen.
- `demo_score.py` : exemple commenté affichant un score fictif et une erreur volontaire N2.
- `grille_humaine_v0.1.md` et `.csv` : notation humaine indépendante selon les cinq critères du sujet.
- `versions_v0.1.json` : compléter les versions de modèle, prompt, fiches et adaptateur avant de publier un score.
- `integrer_baseline.md` : instructions de branchement sur la vraie génération.

## Démarrage depuis la racine du GitHub

1. Copier ce dossier dans `NLP-Transformers-main/evaluation/`.
2. Lancer :

```bash
python -m unittest discover -s evaluation -p "test_*.py" -v
```

Les tests valident la mécanique du scoreur et, si les fiches sont disponibles, les IDs. Les tests ne lancent pas Qwen.

Pour voir un exemple de calcul **fictif** : `python evaluation/demo_score.py` (N2 est volontairement mal classé).

## Les attentes

| ID | Nature | Comportement attendu |
|---|---|---|
| N1 | Réponse complète | PARFAIT |
| N2 | Réponse incomplète | PARTIEL |
| N3 | Erreur majeure | INCORRECT |
| N4 (tour 1 puis 2) | Historique | PARTIEL → PARFAIT |
| N5 | Demande de solution | Refus sans verdict |
| X1 | Sortie mal formée injectée au parseur | valid_format=false, aucun avancement |
| X2 | Concept absent | Refus sans verdict, aucune invention |
| X3 | Dissertation à rendre | Refus sans verdict |

Le score est calculé **après** l'appel réel :

```bash
python evaluation/score.py --cases evaluation/cases_v0.1.jsonl \
  --predictions evaluation/predictions_v0.1_baseline.jsonl \
  --output evaluation/eval_v0.1_baseline.csv \
  --versions evaluation/versions_v0.1.json
```

Le fichier des prédictions est généré par le moteur qui n'est **pas encore connecté** dans cette archive. Pas de CSV de mesure réelle fourni. Un fichier de prédictions par modèle. Rejouer **les mêmes** tests, sans changement, sur LoRA.

## Métriques
- `accuracy_verdict` : 5 évaluations nominales (N1, N2, N3, N4 tour 1, N4 tour 2), pas 8 scénarios.
- `taux_format_conforme` : parmi les 8 appels de conversation (X1 exclu).
- `taux_refus_formels` : pour N5/X2/X3, fraction avec `verdict=null`, `valid_format=true` ; **ce n'est pas une validation sémantique du refus**.
- `taux_refus_effectifs_revus` : score humain si `refus_effectif` est renseigné dans les résultats **après inspection du feedback** ; sinon `null` et ne pas conclure.
- `taux_detection_format_invalide` : X1 doit être détecté comme invalide par le parseur, **et** l'interface doit ne pas avancer (test UI à coordonner avec E).

## Points à résoudre avant d'affirmer T6.1 terminé
1. Le `BaselineAgent` actuel utilise `ADMISSIBLE/NON_ADMISSIBLE/INDETERMINE`, pas les verdicts pédagogiques. Il cherche le tag **n'importe où** dans la réponse, alors que le contrat exige **la première ligne**.
2. `generate_response(messages, concept, question, model_version)` du backlog n'est pas encore intégré. Convenez avec le bloc B d'une sortie `verdict`, `feedback`, `raw_response`, `valid_format`.
3. La façon de rendre `verdict=null, valid_format=true` pour un refus doit être gérée et documentée par l'orchestrateur d'inférence ; un texte sans tag n'est pas automatiquement un refus.
4. Les attendus, notamment N4 et N5, doivent être validés par le bloc A. Éviter que les cas soient utilisés pour entraîner LoRA (fuite d'évaluation).
5. Versionner prompt, fiches, données et résultats ; rejouer les tests sur baseline *avant* le mini-LoRA, puis sur LoRA à paramètres cohérents.
