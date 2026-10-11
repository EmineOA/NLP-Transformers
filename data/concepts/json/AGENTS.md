# data/concepts/json

Un fichier `<id>.json` par concept. Ces fichiers sont générés à partir de `../markdown/` : ne pas les modifier à la main.

Format : contrat 1 (fiche concept), avec en plus `difficulte`, `apres_section`, `indices` et `niveaux`. Tous les textes sont en **texte brut** : sans Markdown (`**`, `*`) ni retour à la ligne.

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
      "indices": ["indice 1 …", "indice 2 …"],
      "niveaux": {
        "parfait":   {"criteres": ["…"], "exemple": "…"},
        "partiel":   {"criteres": ["…"], "exemple": "…"},
        "incorrect": {"criteres": ["…"], "exemple": "…"},
        "refus":     {"criteres": ["…"], "exemple": "…"}
      }
    }
  ]
}
```

| Champ | Sens |
|---|---|
| `matiere` | `mathematiques` ou `philosophie`. |
| `difficulte` | Entier de 0 à 100. |
| `explication` | Le récit, découpé en sections dans l'ordre de lecture. Les paragraphes d'une section sont réunis dans un seul `contenu`. |
| `apres_section` | La question est posée juste après la section n° `apres_section` (numérotée à partir de 1), avant la révélation de sa réponse. |
| `points_cles` | Ce que la réponse doit contenir. |
| `indices` | Exactement 2 indices fixes, valables quelle que soit la réponse : `indices[0]` est affiché après l'échec du 1ᵉʳ essai, `indices[1]` après l'échec du 2ᵉ. |
| `niveaux` | La grille du verdict. `parfait` : **tous** les critères sont présents. `partiel`, `incorrect` et `refus` : **au moins un** des cas décrits. `refus` couvre les demandes interdites, la manipulation et les insultes ; un hors-sujet reste `incorrect`. `exemple` est une réponse-type qui sert à calibrer le modèle, pas une réponse à recopier. |

Il y a 3 questions par concept (2 pour `plier-une-feuille-42-fois`).
