# data/concepts/json

Un fichier `<id>.json` par concept. Ces fichiers sont générés à partir de `../markdown/` : ne pas les modifier à la main.

Format : contrat 1 (fiche concept), avec en plus `difficulte`, `apres_section` et `niveaux`.

```json
{
  "id": "monty-hall",
  "matiere": "mathematiques",
  "concept": "Le problème de Monty Hall (probabilités conditionnelles)",
  "difficulte": 45,
  "explication": [
    {"titre": "Introduction", "contenu": "Texte Markdown de la section…"}
  ],
  "questions": [
    {
      "id": "monty-hall_q1",
      "type": "prédiction",
      "apres_section": 1,
      "enonce": "Tu gardes ou tu changes ? …",
      "points_cles": ["…"],
      "niveaux": {
        "parfait":   {"criteres": ["…"], "exemple": "…"},
        "partiel":   {"criteres": ["…"], "exemple": "…"},
        "incorrect": {"criteres": ["…"], "exemple": "…"}
      }
    }
  ]
}
```

| Champ | Sens |
|---|---|
| `matiere` | `mathematiques` ou `philosophie`. |
| `difficulte` | Entier de 0 à 100. |
| `explication` | Le récit, découpé en sections dans l'ordre de lecture. Le `contenu` est du Markdown simple (`**gras**`, `*italique*`). |
| `apres_section` | La question est posée juste après la section n° `apres_section` (numérotée à partir de 1), avant la révélation de sa réponse. |
| `points_cles` | Ce que la réponse doit contenir. |
| `niveaux` | La grille du verdict. `parfait` : **tous** les critères sont présents. `partiel` et `incorrect` : **au moins un** des cas décrits. `exemple` est une réponse-type qui sert à calibrer le modèle, pas une réponse à recopier. |

Il y a 3 questions par concept (2 pour `plier-une-feuille-42-fois`).
