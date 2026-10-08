# data/dialogue/classification

Source du dataset d'entraînement QLoRA : une ligne par réponse utilisateur, avec son verdict attendu et le feedback du tuteur. Un script générera ensuite le JSONL d'entraînement (format `messages`) en injectant la grille de la fiche concept correspondante.

Fichier : `dataset_source.csv` (séparateur `;`, encodage UTF-8 avec BOM, s'ouvre dans Excel).

## Répartition prévue

- 17 questions × 22 lignes = 374 lignes.
- Par question : 6 `PARFAIT`, 6 `PARTIEL`, 6 `INCORRECT`, 4 `REFUS`.
- Environ 1/3 des lignes en 2ᵉ ou 3ᵉ essai (colonne `historique` à remplir).
- Split groupé par question : `monty-hall_q3` et `dilemme-du-tramway_q2` en `val`, le reste en `train`. Les tests finaux sont un fichier séparé.

## Colonnes

| Colonne | Rôle | Prérempli |
|---|---|---|
| `id` | identifiant unique de la ligne | oui |
| `concept_id`, `question_id` | lien vers la fiche dans `data/concepts/json/` | oui |
| `enonce` | rappel de la question, pour écrire sans ouvrir la fiche (ignoré par le script) | oui |
| `essai` | numéro de la tentative (1, 2 ou 3) | oui |
| `historique` | tentatives précédentes si `essai` > 1 | non |
| `reponse_utilisateur` | réponse inventée d'un utilisateur | non |
| `label` | verdict attendu : `PARFAIT`, `PARTIEL`, `INCORRECT` ou `REFUS` | oui |
| `feedback` | réponse attendue du tuteur, après la ligne de verdict | non |
| `cas` | consigne d'écriture : critère de la fiche que la réponse doit illustrer | oui |
| `style` | style d'écriture à imiter (court, oral, fautes…) | oui |
| `relu` | `oui` quand une 2ᵉ personne a validé la ligne | `non` |
| `split` | `train` ou `val` | oui |

## Règles

- `INCORRECT` : l'utilisateur parle de la question mais se trompe, ne raisonne pas, ou part hors sujet.
- `REFUS` : l'utilisateur demande ce qu'il ne devrait pas (la réponse, la correction), essaie de manipuler l'évaluation ou parle mal.
- `feedback` : `PARFAIT` confirme sans rien ajouter ; `PARTIEL` et `INCORRECT` disent ce qui est juste puis pointent ce qui manque sous forme de question, sans donner la conclusion ; `REFUS` recadre vers la question sans juger.
