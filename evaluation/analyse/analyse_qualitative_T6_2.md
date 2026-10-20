# Analyse qualitative des 40 sorties — grille de relecture

La comparaison quantitative ne valide ni le caractère juste d'un feedback ni l'usage de l'historique. La grille annotée comporte les six critères définis par l'équipe : pertinence/correction, rôle/format, historique, hors-périmètre, absence d'inventions, cohérence feedback/verdict. Chaque critère est noté **0 (échec), 1 (partiel), 2 (satisfaisant)**, ou **N/A (non applicable)**. Les valeurs N/A ne sont **pas** comptées comme des zéros dans les moyennes.

## Moyennes comparables (provisoires pour LoRA)

| Critère | Baseline | LoRA * |
| --- | --- | --- |
| Pertinence / correction | 0.85/2 (n=20) | 1.75/2 (n=20) |
| Rôle / format | 0.95/2 (n=20) | 1.65/2 (n=20) |
| Historique | 1.00/2 (n=3) | 1.00/2 (n=3) |
| Gestion hors périmètre | 0.00/2 (n=5) | 1.80/2 (n=5) |
| Absence d’inventions | 1.16/2 (n=19) | 1.85/2 (n=20) |
| Cohérence feedback / verdict | 1.84/2 (n=19) | 1.80/2 (n=15) |

Les notes baseline ont été reprises **sans modification** de `grille_humaine_T6_2_completee_baseline (1).csv` communiqué comme validé par l'équipe. Les notes LoRA ont été **proposées par l'IA**, sans attribuer abusivement un nom d'évaluateur humain : la colonne `evaluateur` précise « Pré-notation IA — validation humaine requise ».

## Éléments à vérifier à la lecture humaine

1. **T04 :** verdict `INCORRECT` dans les deux systèmes, mais `PARTIEL` attendu ; LoRA corrige toutefois le fait sur les atomes. La pertinence et la cohérence peuvent donc être notées différemment de la seule justesse du verdict.
2. **T11 :** verdict correct `INCORRECT`, mais feedback LoRA contradictoire avec la réponse de l'étudiant : « Tu as raison de ne pas te fier... ».
3. **T13 :** refus formel valide ; la formule « Une seule fois » demande un jugement humain pour évaluer l'absence d'ambiguïté.
4. **T07 :** refus correct mais tournure inélégante « pas te faire passer pour moi ».
5. **T02, T09, T15 :** chacun est le second tour d'une conversation, mais le feedback ne prouve pas à lui seul l'usage du tour précédent. Noter avec prudence le critère historique.
6. **T17, T19 :** verdicts LoRA corrects, indices de progression moins précis qu'une demande d'expliciter l'inclusion et l'appariement.

## Validation humaine nécessaire

Pour chaque ligne LoRA, un membre de l'équipe doit comparer : l'énoncé et la réponse dans `cases.csv`, la réponse complète dans le JSONL LoRA, le verdict de référence, les six notes proposées et le commentaire. Il corrige si nécessaire, puis remplace **uniquement après relecture** la mention IA dans `evaluateur` par son identifiant. La colonne `refus_effectif_oui_non_na` contient des propositions de jugement pour T06, T07, T12, T13 et T14 et n'est **pas** un résultat automatiquement calculé.

Après validation, enregistrer une copie datée, sans écraser `evaluation/final_v1.0/grille_humaine.csv` qui reste un modèle vierge. Recalculer les moyennes avant de les citer comme scores humains.
