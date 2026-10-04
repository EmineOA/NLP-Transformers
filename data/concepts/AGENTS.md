# data/concepts

6 fiches concept au format JSON, une par fichier (`<id>.json`) : 3 en mathématiques, 3 en philosophie.

Format : contrat 1 (fiche concept) + `difficulte` (0-100) + `apres_section` dans chaque question (la question est posée juste après la section n° `apres_section` de `explication`, avant la révélation de la réponse) + `niveaux` dans chaque question : `{parfait, partiel, incorrect}`, chacun avec `criteres` (liste) et `exemple` (réponse-type), pour guider le verdict du LLM. Règle de lecture : `parfait` = tous les critères ; `partiel` et `incorrect` = au moins un des cas.

Fichiers générés automatiquement à partir des fiches Markdown : ne pas les modifier à la main.
