# BrainMaxxing — Bloc B : moteur LLM et baseline conversationnelle

> Documentation consolidée du bloc B (E2 et E3) et de son raccordement au bloc A.  
> Périmètre : **modèle de base sans entraînement**, observation du Transformer, génération conversationnelle, utilisation des fiches pédagogiques, comparaison de prompts et tests de développement.

## 1. Objet et état du bloc

Le bloc B met en œuvre une baseline conversationnelle qui évalue les réponses écrites d'un étudiant. **Le bloc A définit le contenu pédagogique et les critères de classement ; le bloc B les transmet au modèle et interprète le format de ses sorties.** Le moteur B ne modifie pas les règles du bloc A et ne pilote pas la navigation de l'interface.

| Epic / US | Objet | Mise en œuvre |
|---|---|---|
| E2 · US 2.1 | Tokens, logits, softmax | `src/llm/model.py`, `experiments/tokens_logits.ipynb`, notebook académique |
| E2 · US 2.2 | Greedy, température, top-k et top-p | `src/llm/decoding.py`, `experiments/decoding.ipynb`, notebook académique |
| E3 · US 3.1 | Baseline avec prompt, chat template et historique | `src/llm/prompts.py`, `src/llm/generation.py`, `src/agents/` |
| E3 · US 3.2 | Comparaison de deux prompts | `scripts/compare_prompts.py`, `experiments/results/` |

**Modèle observé lors des exécutions :** `Qwen/Qwen2.5-3B-Instruct`, chargé avec `transformers`, exécuté sur CUDA. Le modèle de base est utilisé en mode inférence (`eval`), **sans mise à jour de ses poids**. Le backend choisit CUDA si disponible, puis MPS, puis CPU.

### Validation réalisée le 16 octobre 2026

| Vérification | Résultat constaté | Ce que cela prouve |
|---|---|---|
| Tests techniques du bloc B | **13/13** réussis | Contrats, parsing, sélection des fiches, historique borné et erreurs d'entrée |
| Validation du corpus | **6 fiches / 17 questions** | Chargement et construction des prompts pour toutes les questions |
| Smoke test réel avec Qwen | **2/2** résultats conformes aux attentes du script | Chemin JSON → prompt → Qwen → parseur fonctionnel sur ces deux cas |
| Comparaison `criteria` | **4/9** corrects ; format **8/9** | Score du jeu de développement, pas une mesure finale |
| Comparaison `fewshot` | **6/9** corrects ; format **8/9** | Configuration retenue *provisoirement* sur ce jeu de développement |

Les taux ci-dessus proviennent des sorties d'exécution enregistrées localement. **Ils ne constituent pas une évaluation indépendante**, et les labels des cas de développement restent à faire valider avec le bloc A. Une baseline opérationnelle peut encore produire des verdicts pédagogiquement erronés.

---

## 2. Organisation des fichiers

```text
BrainMaxxing/                     # ou autre nom de la racine du dépôt
├── data/
│   └── concepts/
│       ├── markdown/              # sources de rédaction : bloc A
│       └── json/                  # fiches générées : bloc A, lues par B
├── src/
│   ├── agents/
│   │   ├── concepts.py            # chargement et validation des fiches
│   │   ├── baseline.py            # classe BaselineAgent + parsing
│   │   ├── history.py             # historique conversationnel borné
│   │   └── inference.py           # API commune generate_response(...)
│   └── llm/
│       ├── __init__.py
│       ├── model.py               # modèle, tokenizer, device
│       ├── decoding.py            # méthodes de décodage explicites
│       ├── generation.py          # génération via chat template
│       └── prompts.py             # prompts pédagogiques criteria / fewshot
├── experiments/
│   ├── bloc_b_academique.ipynb    # synthèse des étapes 2 et 3 du sujet
│   ├── tokens_logits.ipynb        # expérience détaillée Transformer
│   ├── decoding.ipynb             # expérience détaillée décodage
│   ├── dev_cases_bloc_b.jsonl     # jeu de développement du bloc B
│   └── results/                   # sorties de la comparaison
├── scripts/
│   ├── test_integrated_smoke.py   # deux requêtes réelles sur Qwen
│   ├── compare_prompts.py        # comparaison des deux stratégies
│   └── run_evaluation_v01.py     # adaptateur vers le bloc D (ultérieur)
├── tests/
│   └── test_bloc_b.py             # tests techniques sans chargement de Qwen
└── README_BLOC_B.md
```

Les notebooks `baseline.ipynb` et `prompt_comparison.ipynb` des premiers essais peuvent être conservés dans les **archives d'exploration** : ils testent un cas fictif d'admissibilité et ne sont plus les références métier BrainMaxxing. Il en va de même des scripts exploratoires ponctuels (`test_batch.py`, `test_history.py`, etc.) lorsque leurs enseignements sont conservés.

**Principe d'organisation :** les fonctions réutilisables résident dans `src/` ; les notebooks présentent les expériences ; les scripts exécutent les protocoles ; `results/` archive les sorties de développement.

---

## 3. Raccordement avec le bloc A

### 3.1. Fiches pédagogiques et source de vérité

Le bloc A fournit six fiches sur deux matières :

| Matière | Concepts / fichiers JSON | Questions |
|---|---|---:|
| Philosophie | `bateau-de-thesee`, `dilemme-du-tramway`, `doute-de-descartes` | 9 |
| Mathématiques | `infinis-plus-grands-que-l-infini`, `monty-hall`, `plier-une-feuille-42-fois` | 8 |
| **Total** | **6 fiches** | **17** |

Les fichiers `data/concepts/markdown/` sont les **sources éditoriales**. Les fichiers `data/concepts/json/` sont des artefacts **générés** : ne pas les modifier manuellement. Toute correction de contenu ou de grille se décide avec le bloc A, s'effectue dans les Markdown sources puis est répercutée dans les JSON.

Structure utilisée :

```text
Fiche : id, matiere, concept, difficulte, explication[], questions[]
Question : id, type, apres_section, enonce, points_cles[], indices[], niveaux
Niveaux : parfait, partiel, incorrect, refus
Pour chaque niveau : criteres[], exemple
```

Les types de questions sont propres aux fiches (`prédiction`, `argumentation`, `position + critère`, etc.) : le moteur ne suppose pas que chaque question correspond systématiquement à une définition ou à un exercice à réponse unique.

- `apres_section` délimite le cours accessible **avant** la question. Le prompt ne doit pas inclure les sections futures.
- `niveaux[*].criteres` fournit la grille de classement propre à la question ; elle prime sur un jugement général du modèle.
- `points_cles` sert de référence pédagogique sans remplacer la grille des niveaux.
- `niveaux[*].exemple` peut servir d'étalon au prompt *few-shot*, mais **pas de test indépendant**.
- `indices` contient deux aides prévues par le bloc A ; l'affichage après les essais relève du parcours piloté par Gradio.

### 3.2. Charger un concept et sélectionner une question

```python
from src.agents.concepts import load_concept, load_question

fiche = load_concept("monty-hall")
question = load_question(fiche, "monty-hall_q1")

print(fiche["concept"])
print(question["enonce"])
print(question["niveaux"]["partiel"]["criteres"])
```

`load_concept()` vérifie l'identifiant, la présence et la structure des fiches ainsi que la cohérence des sections et des questions. `load_question()` recherche la question dans la fiche sélectionnée. Une question d'un autre concept n'est pas acceptée par l'API d'inférence.

### 3.3. Flux de bout en bout

```text
data/concepts/markdown/   [source de rédaction : bloc A]
          │ génération par le bloc A
          ▼
data/concepts/json/       [fiches utilisées par le moteur]
          │
          ▼
load_concept() + load_question()
          │
          ▼
build_pedagogical_prompt(fiche, question, version=...)
          │ + messages user/assistant + historique borné
          ▼
tokenizer.apply_chat_template(...)
          │
          ▼
Qwen2.5-3B-Instruct → model.generate(...)
          │
          ▼
parse_pedagogical_response(...)
          │
          ▼
{verdict, feedback, raw_response, valid_format}
          │
          ├── Bloc D : évaluation
          ├── Bloc E : Gradio
          └── Bloc C : comparaison future avec LoRA
```

---

## 4. API d'inférence et contrat de sortie

L'interface commune du moteur se trouve dans `src/agents/inference.py` :

```python
from src.agents.concepts import load_concept, load_question
from src.agents.inference import generate_response

fiche = load_concept("monty-hall")
question = load_question(fiche, "monty-hall_q1")

resultat = generate_response(
    messages=[{
        "role": "user",
        "content": "Je garde, car il reste deux portes, donc 50/50."
    }],
    concept=fiche,
    question=question,
    model_version="baseline",
    prompt_version="fewshot",
)
print(resultat)
```

Champs du résultat :

```python
{
    "verdict": "PARTIEL",                  # ou PARFAIT / INCORRECT / None
    "feedback": "Retour pédagogique...",
    "raw_response": "[VERDICT: PARTIEL]\nRetour pédagogique...",
    "valid_format": True,
}
```

Cet objet est **un exemple de structure**, pas une promesse que Qwen choisira toujours le verdict attendu.

### Règles de parsing

| Sortie brute | Interprétation |
|---|---|
| `[VERDICT: PARFAIT]` suivi d'un feedback | `PARFAIT`, format valide |
| `[VERDICT: PARTIEL]` suivi d'un feedback | `PARTIEL`, format valide |
| `[VERDICT: INCORRECT]` suivi d'un feedback | `INCORRECT`, format valide |
| `[HORS_PERIMETRE]` suivi d'un feedback | `verdict=None`, format valide **selon la convention provisoire** |
| Marqueur manquant, dupliqué, déplacé ou sans feedback | `verdict=None`, `valid_format=False` |

Le parseur exige **un marqueur unique en première ligne et un feedback non vide**. Il ne modifie pas le verdict produit pour le rendre conforme à la grille. Une erreur du modèle est donc visible dans les résultats de l'évaluation.

Le niveau `refus` existe dans les fiches du bloc A, mais sa traduction textuelle `[HORS_PERIMETRE]` puis `verdict=None` doit encore être **approuvée collectivement**. De même, selon les fiches A actuelles, une réponse simplement hors sujet relève de `INCORRECT`, alors qu'une demande de solution directe ou de manipulation est un refus.

### Gestion du périmètre

L'API refuse de façon contrôlée un concept absent du corpus ou une question n'appartenant pas à la fiche. Elle ne demande pas au modèle d'inventer une explication pour une fiche introuvable. Les messages `system` externes ne sont pas acceptés : le prompt système est construit par le moteur à partir de la grille A.

### Historique

- Les messages suivent les rôles `user` et `assistant` ; le dernier message reçu doit provenir de l'étudiant.
- `generate_response()` ne possède **aucun historique caché** : l'appelant fournit les messages.
- Par défaut, le moteur conserve au plus **trois échanges complets antérieurs**, plus la nouvelle demande utilisateur.
- `BaselineAgent` offre une interface pratique avec `ask()`, `set_question()` et `reset()` ; il s'appuie sur le même moteur pédagogique.

**Limite métier connue :** trois échanges récents ne garantissent pas de conserver la règle formulée à Q1 pour une Q2 ou Q3 philosophique après de nombreux essais. Le bloc A doit préciser les réponses précédentes à maintenir ; le bloc C et le futur parcours Gradio devront respecter la même politique.

`model_version="lora"` n'est pas branché dans ce bloc : ce choix lève volontairement `NotImplementedError`. Le backend LoRA relève du bloc C et doit respecter le même contrat de sortie.

---

## 5. Comprendre le Transformer — E2

### 5.1. Modèle et tokenisation

`src/llm/model.py` charge modèle causal, tokenizer et device. Dans les observations de tokenisation, le texte `Paris est la capitale de` a été décomposé en **six tokens** ; les token IDs et leurs représentations textuelles ont été inspectés.

La tokenisation convertit un texte en identifiants discrets. Une couche d'**embeddings** associe ensuite chaque identifiant à une représentation vectorielle que le Transformer transforme en tenant compte du contexte.

### 5.2. Logits, attention causale, softmax et unembedding

Pour une entrée de six tokens, l'une des exécutions documentées a affiché des logits de forme **`(1, 6, 151936)`** : batch de taille 1, six positions, et un score pour chacun des 151 936 tokens du vocabulaire à chaque position.

- L'**attention causale** empêche une position de lire les tokens futurs ; le modèle prédit le prochain token à partir du préfixe déjà disponible.
- L'**unembedding** projette la représentation finale en scores pour les tokens du vocabulaire : les *logits*.
- La distribution du prochain token est obtenue en appliquant `softmax` aux logits de **la dernière position**, `outputs.logits[:, -1, :]`.
- `argmax` sélectionne le token le plus probable ; la génération autorégressive ajoute ce token et recommence l'opération.

Ces opérations sont illustrées dans `experiments/tokens_logits.ipynb` et dans `experiments/bloc_b_academique.ipynb`.

### 5.3. Stratégies de décodage

`src/llm/decoding.py` met en œuvre :

```python
apply_temperature(logits, temperature)
apply_top_k(logits, k)
apply_top_p(logits, p)
sample_next_token(logits, temperature, top_k, top_p)
generate(model, tokenizer, prompt, device, ...)
```

Ordre des transformations dans l'expérience :

```text
logits → division par température → top-k → top-p
       → softmax → torch.multinomial → token suivant
```

| Méthode | Principe et effet |
|---|---|
| Greedy (`argmax`) | Prend systématiquement le token de logit maximal |
| Température | `logits / T` : `T < 1` concentre ; `T > 1` diffuse la distribution |
| Top-k | Ne conserve que les `k` candidats les mieux classés ; les autres sont masqués |
| Top-p | Retient le plus petit ensemble dépassant le seuil de probabilité cumulée `p` ; nombre variable de candidats |
| Sampling | Tire un token selon la distribution avec `torch.multinomial()` |

Exemples **observés dans les expériences antérieures** : un filtrage `top-k` avec `k=5` ne conservait que cinq candidats ; `top-p=0.9` en conservait 52 pour la distribution étudiée. Ce dernier effectif dépend fortement du contexte et n'est pas une propriété fixe de top-p.

### 5.4. Génération conversationnelle

`src/llm/generation.py` applique le chat template natif de Qwen :

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

La baseline exploite `model.generate(...)` avec `do_sample=False` et décode **uniquement les nouveaux tokens** : la réponse ne contient donc pas une copie du prompt d'entrée. Pour les comparaisons de développement, la génération déterministe réduit la variabilité entre configurations. Le paramètre `max_new_tokens=160` est celui du point d'entrée commun du bloc B.

---

## 6. Prompts de la baseline — E3

`src/llm/prompts.py` construit deux configurations pédagogiques pour **les mêmes fiches et les mêmes messages** :

- **`criteria`** : énoncé, cours accessible, points de référence, grille et règles de feedback.
- **`fewshot`** : même configuration, enrichie des réponses étalons `niveaux[*].exemple` fournies par le bloc A.

Les critères des fiches priment sur les opinions générales du modèle. En philosophie, une position « oui » ou « non » peut être recevable dès lors qu'elle satisfait la grille argumentative ; le moteur doit juger la justification, non imposer une opinion unique.

Les sorties doivent commencer par un verdict puis fournir un feedback court, sans recopier la solution ni les exemples étalons. Les sorties non conformes et les erreurs de classement sont conservées telles quelles.

### Expérience réelle de comparaison

Commande exécutée :

```powershell
python scripts/compare_prompts.py
```

| Configuration | Verdicts et formats corrects | Conformité du format | Nombre de cas |
|---|---:|---:|---:|
| `criteria` | 4/9 (**44,44 %**) | 8/9 (**88,89 %**) | 9 |
| `fewshot` | 6/9 (**66,67 %**) | 8/9 (**88,89 %**) | 9 |

Avec `--limit 2`, la comparaison donnait au contraire 1/2 pour `criteria` et 0/2 pour `fewshot`. Le classement s'est inversé après l'élargissement à neuf cas : **la variante few-shot est retenue provisoirement**, sans extrapoler cette différence à une évaluation générale.

Le refus est la principale erreur de format commune aux deux configurations sur cette exécution. Les tests ont aussi mis en évidence des erreurs d'application de la grille philosophique (par exemple sur le bateau de Thésée). Le parseur ne les corrige pas : elles font partie du comportement réel de la baseline.

### Artefacts générés dans `experiments/results/`

| Fichier | Utilité |
|---|---|
| `comparaison_prompts_bloc_b.csv` | Détail par réponse : prompt, attendu, prédit, format et texte généré |
| `comparaison_prompts_resume.json` | Synthèse des métriques de développement |
| `baseline_reference_dev.jsonl` | Réponses de la configuration sélectionnée pour les comparaisons futures |

Conserver les résultats avec l'identifiant du modèle, la version du prompt, les fiches, les paramètres et le commit Git. Les cas ayant servi à choisir un prompt ne doivent pas être présentés comme un **jeu final indépendant**.

---

## 7. Vérifications et commandes

Exécuter depuis la **racine du dépôt**, où `src/`, `data/`, `scripts/` et `tests/` sont côte à côte. Le nom de cette racine peut être `BrainMaxxing` ou `test`, selon l'installation locale.

### Tests techniques sans GPU

```powershell
python -m unittest discover -s tests -p "test_bloc_b.py" -v
```

**Résultat observé : 13 tests réussis, zéro échec.** Ces tests vérifient le logiciel (y compris avec des réponses de modèle simulées), pas l'exactitude pédagogique de Qwen.

### Validation du corpus

```powershell
python test_concepts_all.py
```

**Résultat observé : six fiches et dix-sept questions validées.** Si ce script est déplacé dans `tests/`, adapter la commande en conséquence. Cette validation structurelle ne prouve pas que toutes les réponses possibles seront bien évaluées.

### Smoke test avec Qwen

```powershell
python scripts/test_integrated_smoke.py
```

**Résultat observé :** `monty-hall_q1` donne `PARTIEL` sur « Je garde, car il reste deux portes donc 50/50 » et renvoie un refus formel valide pour une demande de solution immédiate. Ces deux cas illustrent un raccordement effectif, sans garantir le comportement de l'ensemble du corpus.

### Comparaison des prompts

```powershell
python scripts/compare_prompts.py --limit 2
python scripts/compare_prompts.py
```

Le second appel alimente les fichiers `experiments/results/` décrits plus haut.

### Notebook académique

Ouvrir `experiments/bloc_b_academique.ipynb` et exécuter les cellules dans l'ordre. Les notebooks `tokens_logits.ipynb` et `decoding.ipynb` constituent les expériences détaillées complémentaires. Les résultats réels et leurs interprétations sont conservés dans le notebook académique exécuté.

### Intégration ultérieure du bloc D

Une fois le dossier `evaluation/` disponible dans le même dépôt :

```powershell
python scripts/run_evaluation_v01.py
python evaluation/score.py --cases evaluation/cases_v0.1.jsonl --predictions evaluation/predictions_v0.1_baseline.jsonl --output evaluation/eval_v0.1_baseline.csv --versions evaluation/versions_v0.1.json
```

Avant une évaluation, renseigner `evaluation/versions_v0.1.json` avec les versions exactes. Les tests de validation des composants ne sont pas une mesure de Qwen ; de même, **l'évaluation finale doit utiliser au moins vingt requêtes indépendantes** et rejouer les mêmes cas sur la baseline et le LoRA. Les verdicts attendus ne doivent jamais être injectés dans le prompt du modèle lors du score.

---

## 8. Décisions communes encore ouvertes

| Sujet | Convention actuelle | Décision collective attendue |
|---|---|---|
| **Refus** | Niveau `refus` dans A → `[HORS_PERIMETRE]` + feedback → `verdict=None` | Approuver formellement le marqueur brut et la traduction de sortie |
| **Hors sujet** | `INCORRECT` selon les fiches A | Confirmer les frontières avec le refus pour les cas ambigus |
| **Historique entre questions** | Trois échanges récents maximum | Décider comment conserver la règle de Q1 utile à Q2/Q3 |
| **Progression et indices** | `apres_section` et `indices` fournis par A ; navigation pilotée hors B | Confirmer la politique de progression et d'affichage après les essais |
| **Critères des verdicts** | Les `niveaux` de chaque fiche priment sur l'opinion de Qwen | Valider avec A les cas ambigus et les labels du jeu de développement |

Ces décisions doivent ensuite être appliquées **de manière cohérente** au moteur d'inférence, au dataset du bloc C et au parcours Gradio du bloc E.

---

## 9. Dépendances et reproductibilité

Dépendances principales mentionnées dans la documentation initiale :

```text
torch
transformers
accelerate
safetensors
pandas
jupyter
pytest
```

Les scripts de test techniques utilisent également la bibliothèque standard (`unittest`). Le modèle peut être téléchargé depuis Hugging Face lors de son premier chargement. L'exécution locale testée utilise CUDA ; elle n'implique pas que la même mémoire ou durée soit obtenue sur un autre environnement.

Pour reproduire une expérience et la comparer plus tard à LoRA, enregistrer **modèle de base, version des fiches, configuration du prompt, paramètres de génération, version du code et sorties brutes**.

---

## 10. Périmètre et historique du développement

**Le bloc B couvre les étapes 2 et 3 du sujet** : observer le Transformer et produire une baseline sans fine-tuning. Ne font pas partie de cette livraison : création complète des dialogues d'entraînement, entraînement LoRA, scores comparatifs finaux baseline/LoRA, interface Gradio et rapport général.

Les premiers essais du projet utilisaient un exemple fictif d'admissibilité (Alice, Bob, etc.), avec les labels `ADMISSIBLE`, `NON_ADMISSIBLE` et `INDETERMINE`. Ces fonctions subsistent uniquement pour la **compatibilité historique** des anciens notebooks et ne constituent pas le contrat pédagogique. L'ancienne comparaison `V1_simple` / `V2_strict` évaluait surtout le respect du format sur six cas d'admissibilité ; elle a été remplacée, pour BrainMaxxing, par `criteria` / `fewshot` sur les fiches du bloc A.

Une version antérieure de `PROJECT_CONTEXT_PART2.md` mentionnait également `Qwen/Qwen3-0.5B-Instruct` comme modèle par défaut. Cette mention est **obsolète** au regard du code et des exécutions utilisées pour ce bloc, qui s'appuient sur `Qwen/Qwen2.5-3B-Instruct`.

**Règles techniques conservées depuis les premières expérimentations :** code réutilisable dans `src/`, chat template natif, génération déterministe pour les comparaisons, séparation entre erreur de format et erreur de raisonnement, historique borné et sauvegarde systématique des résultats expérimentaux.
