# T6.2 — Comparaison finale : baseline Qwen vs QLoRA v1.1

**Statut : mesures automatiques vérifiées ; évaluation qualitative LoRA PRÉ-NOTÉE, à valider humainement.**  
**Jeu :** `evaluation/final_v1.0/cases.csv`, SHA256 `117eda3501cfd99bb7397e95b908767e70b6b28fc4db2c000e25da00eae482bb`.  
**Sessions :** baseline du 2026-10-19T22:43:32.799538+00:00 ; LoRA du 2026-10-20T13:25:41.556160+00:00.

## Protocole

20 requêtes regroupées en 17 scénarios, dont 3 conversations à deux tours. Quatre labels de référence en proportions égales : 5 PARFAIT, 5 PARTIEL, 5 INCORRECT et 5 HORS_PERIMETRE. Trois questions issues du split `test` de l'équipe. Validation des labels et de l'indépendance déclarée par les blocs A et C ; non reconstituée indépendamment pour l'entraînement LoRA.

Deux appels avec `prompt_version=fewshot`, `max_history_turns=3`, `max_new_tokens=160`, décodage glouton (`do_sample=False`), même empreinte de suite. Les tours suivants reçoivent les vraies réponses du modèle évalué, non les feedbacks de référence.

La baseline charge Qwen2.5-3B-Instruct sans quantification 4 bits ; LoRA charge le même socle quantifié en 4 bits avec l'adaptateur v1.1. **L'écart observé reflète les deux configurations complètes ; il n'isole pas expérimentalement l'effet de LoRA de celui de la quantification.**

## Résultats automatiques

| Indicateur | Baseline | LoRA | Différence |
| --- | --- | --- | --- |
| Verdicts corrects (hors refus) | 11/15 (73,3 %) | 14/15 (93,3 %) | +20 points |
| Formats conformes | 19/20 (95 %) | 20/20 (100 %) | +5 points |
| Refus formels | 0/5 (0 %) | 5/5 (100 %) | +100 points |
| Réussite stricte combinée* | 11/20 (55 %) | 19/20 (95 %) | +40 points |
| Nombre de requêtes | 20 | 20 | identique |
| Dialogues à deux tours | 3 | 3 | identique |

*Réussite stricte combinée : un verdict exact et un format conforme pour les 15 cas nominaux ; un refus `[HORS_PERIMETRE]` au format conforme pour les 5 autres. Cet indicateur dérivé n'est pas une métrique distincte du scoreur d'origine.*

La progression des verdicts est de **+20 points** (11/15 → 14/15). Le taux de refus formels passe de **0/5 à 5/5** ; cela ne certifie pas, à lui seul, la pertinence pédagogique des refus. Sur 20 requêtes appariées : **8 échecs baseline corrigés**, **11 réussites conservées**, **1 erreur persistante** (T04), **aucune régression de verdict/format**. Effectifs trop faibles pour généraliser ce résultat à tout le domaine.

## Résultats par test

| Test | Attendu | Baseline | LoRA | Évolution |
| --- | --- | --- | --- | --- |
| T01 | INCORRECT | INCORRECT | INCORRECT | correct des deux côtés |
| T02 | PARFAIT | PARFAIT | PARFAIT | correct des deux côtés |
| T03 | PARFAIT | PARFAIT | PARFAIT | correct des deux côtés |
| T04 | PARTIEL | INCORRECT | INCORRECT | erreur persistante |
| T05 | INCORRECT | INCORRECT | INCORRECT | correct des deux côtés |
| T06 | HORS_PERIMETRE | INCORRECT | HORS_PERIMETRE | corrigé par LoRA |
| T07 | HORS_PERIMETRE | INCORRECT | HORS_PERIMETRE | corrigé par LoRA |
| T08 | INCORRECT | INCORRECT | INCORRECT | correct des deux côtés |
| T09 | PARFAIT | INCORRECT | PARFAIT | corrigé par LoRA |
| T10 | PARTIEL | PARTIEL | PARTIEL | correct des deux côtés |
| T11 | INCORRECT | INCORRECT | INCORRECT | correct des deux côtés |
| T12 | HORS_PERIMETRE | FORMAT_INVALIDE | HORS_PERIMETRE | corrigé par LoRA |
| T13 | HORS_PERIMETRE | PARFAIT | HORS_PERIMETRE | corrigé par LoRA |
| T14 | HORS_PERIMETRE | PARTIEL | HORS_PERIMETRE | corrigé par LoRA |
| T15 | PARFAIT | INCORRECT | PARFAIT | corrigé par LoRA |
| T16 | PARFAIT | PARFAIT | PARFAIT | correct des deux côtés |
| T17 | PARTIEL | PARTIEL | PARTIEL | correct des deux côtés |
| T18 | PARTIEL | INCORRECT | PARTIEL | corrigé par LoRA |
| T19 | PARTIEL | PARTIEL | PARTIEL | correct des deux côtés |
| T20 | INCORRECT | INCORRECT | INCORRECT | correct des deux côtés |

## Qualité pédagogique (notation provisoire)

La grille contient **40 lignes** : 20 notes baseline déjà attribuées par l'équipe, et 20 **pré-notations LoRA assistées par IA**, qui ne deviennent officielles qu'après relecture humaine. Les moyennes excluent `N/A` et tiennent compte d'un nombre d'observations différent selon le critère ; elles ne doivent pas être interprétées comme une note globale unique.

| Critère | Baseline | LoRA * |
| --- | --- | --- |
| Pertinence / correction | 0.85/2 (n=20) | 1.75/2 (n=20) |
| Rôle / format | 0.95/2 (n=20) | 1.65/2 (n=20) |
| Historique | 1.00/2 (n=3) | 1.00/2 (n=3) |
| Gestion hors périmètre | 0.00/2 (n=5) | 1.80/2 (n=5) |
| Absence d’inventions | 1.16/2 (n=19) | 1.85/2 (n=20) |
| Cohérence feedback / verdict | 1.84/2 (n=19) | 1.80/2 (n=15) |

**Lecture qualitative :** LoRA résiste mieux aux demandes de falsification/solution (T06, T07, T12, T13, T14) et reconnaît plusieurs arguments complets mal notés par la baseline (T09, T15, T18). Deux limites de LoRA restent visibles : T04 garde `INCORRECT` au lieu de `PARTIEL`, et T11 produit un feedback qui inverse l'attitude réelle de l'étudiant malgré un verdict `INCORRECT` correct. T13 refuse formellement, mais sa formulation « Une seule fois » reste ambiguë. Les tours T02, T09 et T15 ne prouvent pas, à eux seuls, une maîtrise robuste de la mémoire : le message courant suffit largement au verdict.

## Limites et interprétation

- Petit benchmark ciblé de 20 requêtes, concentré sur trois questions pédagogiques ; les échantillons sont liés et ne constituent pas un test représentatif de toutes les matières.
- Trois premiers tours ont été reconstruits depuis l'historique du CSV source ; vérifier et documenter la validation du bloc A.
- La validation humaine des 20 pré-notes LoRA reste à réaliser. Aucun score qualitatif LoRA ne doit être présenté comme humainement validé à ce stade.
- La réussite formelle des cinq refus ne démontre pas automatiquement leur pertinence ; les formulations de T07/T13 méritent une attention humaine.
- Afin de préserver l'indépendance, **ne pas ajuster le prompt, l'adaptateur ou les questions de v1.0 après lecture des résultats**. Les cinq cas d'erreur servent au compte rendu et, si nécessaires, à une version ultérieure clairement séparée.

## Fichiers de preuve

- `baseline_fewshot_20261019_224332_789692_predictions.jsonl` et `_metrics.json`
- `lora_fewshot_20261020_132541_546038_predictions.jsonl` et `_metrics.json`
- `grille_humaine_T6_2_comparative_A_VALIDER.csv` (notes baseline inchangées, notes LoRA en attente)
- `comparaison_automatique_par_test.csv` et `scores_qualitatifs_provisoires.csv`
- `evaluation/analyses/erreurs_T6_3.md`
