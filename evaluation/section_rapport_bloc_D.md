# Proposition de section du rapport — Bloc D, T6.2 et T6.3

## Méthode

L'évaluation compare Qwen2.5-3B-Instruct (baseline) à Qwen2.5-3B-Instruct avec un adaptateur QLoRA v1.1. Les deux configurations reçoivent la même suite figée v1.0 (SHA256 `117eda3501cfd99bb7397e95b908767e70b6b28fc4db2c000e25da00eae482bb`), contenant 20 requêtes, 17 scénarios et trois conversations de deux tours. Les étiquettes de référence comprennent cinq exemples par catégorie : PARFAIT, PARTIEL, INCORRECT et HORS_PERIMETRE. Les requêtes sont issues de trois questions du split `test` du corpus ; l'indépendance et les étiquettes ont été confirmées par les responsables des blocs C et A selon les informations de l'équipe. Le prompt `fewshot`, l'historique maximal de trois tours, la limite de 160 nouveaux tokens et le décodage déterministe ont été conservés. Les verdicts attendus ne sont pas transmis comme labels du tour courant au modèle.

## Résultats

La baseline obtient **11 verdicts exacts sur 15 (73,3 %)**, contre **14 sur 15 (93,3 %)** pour QLoRA (+20 points). Le format est respecté dans **19/20** générations baseline et **20/20** générations QLoRA. Les refus formels conformes passent de **0/5** à **5/5**. Selon un indicateur strict combinant classification et refus formel, la réussite passe de **11/20 (55 %)** à **19/20 (95 %)**. Sur les 20 mêmes requêtes, huit échecs de la baseline sont corrigés, un échec demeure (T04), et aucune régression formelle n'est observée.

## Analyse des erreurs et discussion

Les erreurs baseline révèlent quatre comportements : sous-évaluation de raisonnements complets (T09, T15), difficulté à traiter les réponses partielles (T04, T18), défaut de refus des demandes de solution ou de manipulation (T06, T07, T12, T13, T14), et contradictions ou attributions abusives dans le feedback (T04, T13, T18). L'adaptateur corrige l'essentiel des erreurs formelles, mais **T04** reste noté `INCORRECT` au lieu de `PARTIEL`. Le feedback LoRA sur **T11** interprète à tort l'étudiant comme sceptique à l'égard de son propre raisonnement, montrant qu'un verdict correct n'assure pas un feedback fidèle.

## Limites

Les résultats ne portent que sur 20 requêtes issues de trois questions, avec trois paires multi-tours. Ils ne permettent pas de conclure à une performance générale. Un refus au bon format ne prouve pas automatiquement sa qualité pédagogique ; la revue qualitative reste nécessaire. De plus, le modèle QLoRA est chargé sur un socle quantifié en 4 bits, contrairement à la baseline : cette comparaison ne sépare pas l'effet du fine-tuning de celui de la quantification. Les notes qualitatives de la baseline proviennent d'une grille transmise comme validée ; **les 20 notes qualitatives LoRA jointes sont des pré-notations IA en attente de relecture humaine**. Les conclusions qualitatives chiffrées devront être révisées après validation.

## Traçabilité

Scripts : `evaluation/run_final.py` et `evaluation/score.py`. Sources : `evaluation/final_v1.0/cases.csv`, `selection.json`, `lock.json` ; deux fichiers `*_predictions.jsonl` et `*_metrics.json` conservés avec les résultats de chaque modèle. Analyse des erreurs : `evaluation/analyse/erreurs_T6_3.md`.
