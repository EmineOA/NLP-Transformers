# data/concepts/markdown

Les fiches sources, une par concept : `<matiere>/<id>.md`. Le nom du dossier (`mathematiques` ou `philosophie`, sans accent) donne la matière, et le nom du fichier donne l'`id`.

## Structure d'une fiche, dans cet ordre

~~~
# Titre accrocheur

```yaml
concept: "Nom exact du concept"
difficulte: 45
```

---
## Le récit          ← sous-titres en ### uniquement ; chaque ### = une section du JSON
---
## Corrigé           ← un bloc ### Q<n> — <rappel> par question
---
## Sources et précisions   ← références et chiffres, non exporté
~~~

**Questions**, dans le récit, placées avant la révélation de la réponse :

```
> **À toi de réfléchir — Question 1** *(type)*
> Énoncé…
```

**Corrigé.** Il commence par `> **Comment l'utiliser.** …`. Ensuite, chaque `### Q<n>` contient ces rubriques (`- **Nom** :` puis une liste indentée) :

| Rubrique | Contenu | Exportée en JSON |
|---|---|---|
| `Réponse attendue` (maths) / `Positions valides` (philo) | La bonne réponse, ou les positions défendables | non |
| `Raisonnement attendu` | Les points que la réponse doit contenir | `points_cles` |
| `Erreurs typiques` | Les erreurs fréquentes et ce qu'elles révèlent | non |
| `Réponse excellente` | Ce qui dépasse le niveau parfait | non |
| `Indices` | Exactement 2 indices fixes : l'indice 1 après l'échec du 1ᵉʳ essai, l'indice 2 après l'échec du 2ᵉ | `indices` |
| `Parfait` / `Partiel` / `Incorrect` | Les critères de chaque niveau, suivis d'une ligne `Exemple : « … »` | `niveaux` |

Le gras, l'italique et les retours à la ligne sont retirés à l'export : le JSON ne contient que du texte brut.

## Interdits
- Le caractère `|` n'importe où dans la fiche.
- `##` ou `---` à l'intérieur du récit ou du corrigé : la lecture de la section s'arrêterait là.
- Toute autre clé que `concept` et `difficulte` dans le yaml.
