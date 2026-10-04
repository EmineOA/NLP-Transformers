# data/concepts

Les 6 fiches concept (3 en mathématiques, 3 en philosophie), sous deux formes.

| Dossier | Contenu | Rôle |
|---|---|---|
| `markdown/<matiere>/<id>.md` | La fiche complète : récit, questions, corrigé, indices, niveaux, sources. | **Source.** C'est ici qu'on écrit et qu'on corrige. |
| `json/<id>.json` | Ce que l'agent utilise : explication découpée en sections, questions, grille d'évaluation. | **Généré** à partir des Markdown. Lu par le prompt. |

Règles :
- On corrige le Markdown, jamais le JSON à la main. Le JSON est ensuite régénéré.
- Le même `<id>` (nom du fichier, ex. `monty-hall`) désigne le concept des deux côtés.
- Format détaillé : [`markdown/AGENTS.md`](markdown/AGENTS.md) et [`json/AGENTS.md`](json/AGENTS.md).
