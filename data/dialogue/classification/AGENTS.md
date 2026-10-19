# data/dialogue/classification

Source du dataset d'entraînement QLoRA : une ligne par réponse utilisateur, avec son verdict attendu et le feedback du tuteur. Un LLM générateur remplit les colonnes vides (il renvoie du JSON, fusionné ensuite par un script). Un script générera enfin le JSONL d'entraînement (format `messages`) en injectant la grille de la fiche concept correspondante.

Fichiers :
- `README.md` : méthode de génération et de validation du dataset ;
- `generer_squelette.py` puis `attribuer_generateurs.py` : produisent le squelette (graine fixe, résultat reproductible) ;
- `dataset_source.csv` (séparateur `;`, encodage UTF-8 avec BOM, s'ouvre dans Excel) ;
- `prompt_a_donner.md` : prompt à donner au LLM générateur, une question à la fois ;
- `premier_test.json` : premières lignes générées (bateau de Thésée Q1), pour juger la qualité avant de lancer tout le dataset.

## Répartition

- 17 questions × 59 lignes = 1 003 lignes.
- Par question : 16 `PARFAIT`, 16 `PARTIEL`, 16 `INCORRECT`, 11 `HORS_PERIMETRE`.
- Essais : 493 en essai 1, 255 en essai 2, 255 en essai 3.
- Seules les 174 lignes « Q1, essai 1 » ont un historique vide.
- Split groupé par question (11 / 3 / 3 questions, 649 / 177 / 177 lignes) : `val` = `dilemme-du-tramway_q2`, `monty-hall_q3`, `plier-une-feuille-42-fois_q2` ; `test` = `bateau-de-thesee_q3`, `doute-de-descartes_q1`, `infinis-plus-grands-que-l-infini_q2` ; le reste en `train`. Chaque concept garde au moins une question en `train`.

## Colonnes

| Colonne | Rôle | Prérempli |
|---|---|---|
| `id` | identifiant unique de la ligne | oui |
| `concept_id`, `question_id` | lien vers la fiche dans `data/concepts/json/` | oui |
| `enonce` | rappel de la question actuelle (ignoré par le script) | oui |
| `essai_question` | numéro de l'essai sur la question actuelle (1, 2 ou 3) | oui |
| `labels_historique` | niveau imposé de chaque élément de l'historique (voir plus bas) | oui |
| `historique` | parcours de l'élève avant la réponse actuelle | non |
| `reponse_utilisateur` | réponse de l'élève à la question actuelle | non |
| `label` | verdict attendu : `PARFAIT`, `PARTIEL`, `INCORRECT` ou `HORS_PERIMETRE` | oui |
| `feedback` | réponse attendue du tuteur, après la ligne de verdict | non |
| `cas` | consigne d'écriture : critère de la fiche que la réponse doit illustrer | oui |
| `style` | style d'écriture à imiter (court, hésitant, fautes d'orthographe) | oui |
| `split` | `train`, `val` ou `test` | oui |
| `generateur` | modèle qui écrit la ligne (`claude-opus-5.5` ou `sol-6.1`), tiré au hasard par `attribuer_generateurs.py` | oui |

## Historique

L'historique contient tout le parcours depuis le début du concept :
- **questions précédentes** : seulement le dernier essai de l'élève (sa réponse finale) ;
- **question actuelle** : tous les essais précédents, avec le verdict reçu.

`labels_historique` impose le niveau de chaque élément. Questions précédentes : 60 % `PARFAIT`, 20 % `PARTIEL`, 20 % `INCORRECT` (jamais `HORS_PERIMETRE`). Essais précédents sur la question actuelle : jamais `PARFAIT` (sinon la question serait terminée).

Exemple, ligne `bateau-de-thesee_q3`, essai 2 :

```
labels_historique : bateau-de-thesee_q1:PARFAIT | bateau-de-thesee_q2:PARTIEL | bateau-de-thesee_q3 essai 1:INCORRECT

historique :
[bateau-de-thesee_q1] Oui c'est le même, un objet reste le même tant qu'il n'a jamais cessé d'exister.
[bateau-de-thesee_q2] Le bateau réparé, parce que d'après ma règle il n'a jamais cessé d'exister.
[bateau-de-thesee_q3 – essai 1] Oui j'appuie, c'est plus rapide. → [VERDICT: INCORRECT]
```

## Règles

- `INCORRECT` : l'utilisateur parle de la question mais se trompe, ne raisonne pas, ou part hors sujet.
- `HORS_PERIMETRE` (niveau `refus` des fiches, marqueur `[HORS_PERIMETRE]` du moteur) : l'utilisateur demande ce qu'il ne devrait pas (la réponse, la correction), essaie de manipuler l'évaluation ou parle mal.
- `feedback` : `PARFAIT` confirme sans rien ajouter ; `PARTIEL` et `INCORRECT` disent ce qui est juste puis pointent ce qui manque sous forme de question, sans donner la conclusion ; `HORS_PERIMETRE` recadre vers la question sans juger.
