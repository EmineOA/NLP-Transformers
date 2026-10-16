# generation/

Sorties brutes des LLM générateurs, avant fusion dans `../dataset_source.csv`.

```
generation/
├── manifest.json               # versions : prompt, fiches, modèles, effort, dates, rejets
├── claude-opus-5.5/<question_id>.json
├── sol-6.1/<question_id>.json
└── rapport_validation.csv      # produit par ../valider_dataset.py (ne pas éditer)
```

- Un fichier = les lignes d'une question attribuées à un générateur (colonne `generateur` du squelette).
- Format : tableau JSON de `{id, historique, reponse_utilisateur, feedback}`, exactement comme demandé par `../prompt_a_donner.md`. Aucune autre colonne.
- Préparer un lot : `python data/dialogue/classification/preparer_lot.py <generateur> <question_id>`.
- Valider : `python data/dialogue/classification/valider_dataset.py [--question <question_id>]`. Une ligne `REJET` est regénérée ; une ligne `ALERTE` est relue.
- On ne modifie pas une ligne à la main sans le noter dans `manifest.json`.
