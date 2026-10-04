# BrainMaxxing — contexte technique du bloc B (LLM / baseline)

> Ce fichier sert de contexte rapide pour un développeur ou un LLM qui ouvre le dépôt.  
> Il décrit l'architecture actuelle du **bloc B**, les responsabilités des fichiers, les notebooks d'expérimentation, les contrats utilisés et les points encore en attente.

---

## 1. Objectif du bloc B

Le bloc B construit le moteur LLM **sans entraînement ni fine-tuning**.

Objectifs couverts :

- **US 2.1** — comprendre tokens, logits et softmax ;
- **US 2.2** — comprendre et tester le décodage (`temperature`, `top-k`, `top-p`) ;
- **US 3.1** — construire une baseline conversationnelle ;
- **US 3.2** — comparer plusieurs prompts sur des cas identiques.

Modèle actuellement utilisé :

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Le modèle est chargé localement avec Hugging Face `transformers`.

Le device est sélectionné automatiquement :

```text
CUDA si disponible
MPS si disponible
CPU sinon
```

L'environnement actuellement validé utilise CUDA.

---

## 2. Arborescence

```text
BrainMaxxing/
│
├── experiments/
│   ├── tokens_logits.ipynb
│   ├── decoding.ipynb
│   ├── baseline.ipynb
│   └── prompt_comparison.ipynb
│
├── src/
│   ├── agents/
│   │   ├── baseline.py
│   │   └── history.py
│   │
│   └── llm/
│       ├── __init__.py
│       ├── model.py
│       ├── decoding.py
│       ├── generation.py
│       └── prompts.py
│
├── tests/
│   ├── test_baseline.py
│   └── test_generation.py
│
└── PROJECT_CONTEXT.md
```

Les notebooks servent à **comprendre, expérimenter et valider** les mécanismes.  
Le code réutilisable doit rester dans `src/`.

---

# 3. `src/llm/`

## `model.py`

Responsabilité : charger le modèle, le tokenizer et sélectionner le device.

API utilisée par le reste du projet :

```python
llm = load_model()

llm.model
llm.tokenizer
llm.device
```

Modèle par défaut :

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Le modèle est placé en mode évaluation avec `model.eval()`.  
Il n'y a actuellement **aucun entraînement**.

---

## `decoding.py`

Responsabilité : implémenter explicitement les mécanismes de décodage étudiés dans l'US 2.2.

Fonctions principales :

```python
apply_temperature(logits, temperature)
apply_top_k(logits, k)
apply_top_p(logits, p)
sample_next_token(logits, temperature, top_k, top_p)
generate(model, tokenizer, prompt, device, ...)
```

Pipeline :

```text
logits
  ↓
temperature
  ↓
top-k
  ↓
top-p
  ↓
softmax
  ↓
torch.multinomial()
  ↓
token suivant
```

### Température

```python
scaled_logits = logits / temperature
```

Interprétation :

```text
T < 1  → distribution plus concentrée
T = 1  → distribution originale
T > 1  → distribution plus plate
```

### Top-k

Ne conserve que les `k` tokens ayant les logits les plus élevés.  
Les autres logits sont remplacés par `-inf`.

### Top-p

Conserve le plus petit ensemble de tokens dont la probabilité cumulée atteint le seuil `p`.  
Le nombre de tokens conservés est donc variable.

### Sampling

Le token suivant est échantillonné avec :

```python
torch.multinomial(probabilities, num_samples=1)
```

À distinguer du greedy decoding :

```python
torch.argmax(logits)
```

---

## `generation.py`

Responsabilité : génération conversationnelle utilisée par la baseline.

Cette partie utilise le **chat template natif du tokenizer Qwen** :

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

Puis la génération Hugging Face :

```python
model.generate(...)
```

La baseline utilise actuellement :

```python
do_sample=False
```

La génération est donc déterministe, ce qui facilite les comparaisons expérimentales.

Après la génération, seuls les **nouveaux tokens** doivent être décodés afin de ne pas renvoyer le prompt avec la réponse.

---

## `prompts.py`

Responsabilité : centraliser les prompts système.

Deux variantes sont actuellement utilisées pour l'US 3.2.

### `build_system_prompt_v1`

Prompt volontairement simple et peu contraignant.  
Il sert de baseline de prompting et ne force pas de contrat de sortie strict.

### `build_system_prompt_v2`

Prompt plus contraint.

Il impose notamment :

```text
[VERDICT: ADMISSIBLE]
[VERDICT: NON_ADMISSIBLE]
[VERDICT: INDETERMINE]
```

Il interdit les variantes comme :

```text
[VRAI]
[FAUX]
[OUI]
[NON]
```

Le but est d'obtenir une sortie **parsable automatiquement**.

---

# 4. `src/agents/`

## `history.py`

Responsabilité : stocker un historique conversationnel borné.

Classe principale :

```python
ConversationHistory
```

Méthodes principales :

```python
add_user(...)
add_assistant(...)
get_messages()
clear()
```

Avec :

```python
max_turns=3
```

on conserve au maximum :

```text
3 messages user
+
3 messages assistant
=
6 messages
```

Les messages les plus anciens sont supprimés lorsque la limite est dépassée.

---

## `baseline.py`

Responsabilité : assembler les briques précédentes pour former l'agent baseline.

Classe principale :

```python
BaselineAgent
```

Entrées principales :

```text
model
tokenizer
device
rules
fiche
system_prompt optionnel
max_history_turns
```

Pipeline :

```text
question utilisateur
      ↓
prompt système
      +
règles métier
      +
fiche
      +
historique borné
      ↓
apply_chat_template()
      ↓
model.generate()
      ↓
réponse
      ↓
extract_verdict()
```

### Contrat de verdict

Valeurs autorisées :

```text
ADMISSIBLE
NON_ADMISSIBLE
INDETERMINE
```

Format attendu :

```text
[VERDICT: ADMISSIBLE]
```

Le parseur doit retourner `None` si :

- aucun verdict n'est trouvé ;
- le format est invalide ;
- la valeur n'appartient pas aux valeurs autorisées.

Le résultat de `agent.ask(...)` a la forme logique :

```python
{
    "response": "...",
    "verdict": "ADMISSIBLE"  # ou None
}
```

---

# 5. Notebooks

## `experiments/tokens_logits.ipynb`

Objectif : comprendre le début du pipeline d'un Transformer causal.

La version actuellement sauvegardée contient principalement :

```text
chargement du modèle
      ↓
tokenisation
      ↓
input_ids
      ↓
inspection de la shape et des IDs
```

Exemple utilisé :

```text
Paris est la capitale de
```

Dans l'exécution observée, cette entrée est tokenisée en 6 tokens.

### Attention

La version actuellement inspectée du notebook s'arrête à l'inspection des `input_ids`.

Pour couvrir complètement **US 2.1**, le notebook final devrait aussi conserver :

```text
forward pass
↓
outputs.logits
↓
logits[:, -1, :]
↓
softmax
↓
top tokens
↓
argmax
↓
token suivant
```

Ne pas confondre le contenu actuellement sauvegardé avec l'objectif complet de l'US 2.1.

---

## `experiments/decoding.ipynb`

Objectif : étudier les stratégies de décodage de l'US 2.2.

Le notebook :

1. charge Qwen ;
2. récupère les logits du prochain token ;
3. compare plusieurs températures ;
4. applique `top-k` ;
5. applique `top-p` ;
6. effectue plusieurs tirages ;
7. génère du texte avec plusieurs températures.

Exemple principal :

```text
Paris est la capitale de
```

Shape des logits observée :

```text
[151936]
```

soit un score pour chaque token du vocabulaire.

### Résultats observés

La température vérifie le comportement attendu :

```text
température faible  → distribution plus concentrée
température élevée → distribution plus diffuse
```

Avec :

```python
k=5
```

le notebook conserve exactement 5 tokens possibles.

Avec :

```python
p=0.9
```

l'exécution observée conservait 52 tokens.  
Ce nombre dépend de la distribution.

Le notebook compare aussi des générations avec :

```text
temperature = 0.2
temperature = 0.8
temperature = 1.5
```

Une température élevée donne davantage de variabilité et peut dégrader la cohérence sur ce petit modèle.

---

## `experiments/baseline.ipynb`

Objectif : valider l'agent complet de l'US 3.1.

Le notebook utilise actuellement un **jeu de données fictif** :

```text
Règle :
une personne est admissible si elle a au moins 18 ans.

Fiche :
Nom : Alice
Âge : 22 ans
```

Ce contenu est uniquement un **mock technique**.  
Il ne représente pas les futures données métier du bloc A.

### Tests réalisés

Question :

```text
Alice est-elle admissible ?
```

Résultat validé :

```text
[VERDICT: ADMISSIBLE]
```

Extraction :

```text
ADMISSIBLE
```

Le notebook vérifie aussi explicitement `tokenizer.apply_chat_template(...)`.

Enfin, il teste l'historique borné. Avec `max_history_turns=3`, le résultat observé est :

```text
Nombre de messages : 6
```

### Limite observée

Qwen2.5-0.5B peut produire des réponses sémantiquement faibles ou incohérentes sur certains tours de conversation.

Ce comportement doit être considéré comme une **limite de la baseline**, et non automatiquement comme un bug du code.

---

## `experiments/prompt_comparison.ipynb`

Objectif : comparer expérimentalement deux stratégies de prompting pour l'US 3.2.

Prompts comparés :

```text
V1_simple
V2_strict
```

Les deux prompts sont évalués sur les **mêmes cas**, avec :

```python
do_sample=False
```

afin d'éviter que l'aléatoire fausse la comparaison.

### Dataset temporaire

Six cas sont testés :

```text
Alice  — 22 ans
Bob    — 16 ans
Chloé  — 18 ans
David  — âge inconnu
Emma   — 17 ans
Farid  — 35 ans
```

Ce dataset sert uniquement à valider le protocole d'évaluation.

### Métriques

#### `format_rate`

Mesure la proportion de sorties respectant :

```text
[VERDICT: ...]
```

#### `semantic_accuracy`

Mesure si la décision exprimée est correcte, indépendamment du format.

Il faut distinguer :

```text
erreur de format
```

de :

```text
erreur de raisonnement / décision
```

### Résultats actuels

| Prompt | Semantic accuracy | Format rate |
|---|---:|---:|
| `V1_simple` | 50 % | 0 % |
| `V2_strict` | 50 % | 100 % |

Interprétation :

- `V1_simple` exprime parfois la bonne décision mais ne respecte pas le contrat machine ;
- `V2_strict` impose correctement le format ;
- le prompt strict n'améliore pas ici l'exactitude métier ;
- l'amélioration mesurée porte surtout sur la **conformité de sortie**.

### Stabilité

Trois exécutions identiques avec `V2_strict` et `do_sample=False` ont produit le même verdict.

Résultat :

```text
Réponse stable : True
```

---

# 6. Flux global du bloc B

```text
                 ┌─────────────────────┐
                 │       Bloc A        │
                 │ règles + fiches     │
                 └──────────┬──────────┘
                            │
                     À CONNECTER
                            │
                            ▼
┌──────────────┐   ┌────────────────────┐
│ utilisateur  │──▶│   BaselineAgent    │
└──────────────┘   └─────────┬──────────┘
                             │
                ┌────────────┼─────────────┐
                │            │             │
                ▼            ▼             ▼
             prompt       règles        fiche
                │            │             │
                └────────────┼─────────────┘
                             │
                             ▼
                    historique borné
                             │
                             ▼
                    apply_chat_template
                             │
                             ▼
                  Qwen2.5-0.5B-Instruct
                             │
                             ▼
                       génération
                             │
                             ▼
                     [VERDICT: ...]
                             │
                             ▼
                      extract_verdict
```

---

# 7. Interface attendue avec le bloc A

**Cette partie n'est pas encore figée.**

Le bloc B attend conceptuellement deux éléments :

```text
rules
fiche
```

Ils sont actuellement transmis sous forme de chaînes de caractères.

Exemple temporaire :

```python
rules = """
Une personne est admissible si elle a au moins 18 ans.
"""

fiche = """
Nom : Alice
Âge : 22 ans
"""
```

La forme définitive doit être décidée avec l'équipe responsable du bloc A.

Tant que ce contrat n'est pas défini :

- ne pas coder de dépendance forte à un format métier arbitraire ;
- garder `BaselineAgent` aussi indépendant que possible ;
- considérer les fiches Alice/Bob/etc. comme des fixtures de test uniquement.

---

# 8. État actuel des User Stories

```text
US 2.1 — Tokens / logits / softmax
    mécanismes étudiés
    ⚠ notebook sauvegardé à compléter pour conserver toute la démonstration

US 2.2 — Décodage
    ✅ temperature
    ✅ top-k
    ✅ top-p
    ✅ sampling
    ✅ génération comparative

US 3.1 — Baseline
    ✅ chargement du modèle
    ✅ génération
    ✅ apply_chat_template
    ✅ prompt système
    ✅ historique borné
    ✅ contrat [VERDICT: ...]
    ✅ extraction du verdict

US 3.2 — Comparaison de prompts
    ✅ V1 vs V2
    ✅ cas identiques
    ✅ format_rate
    ✅ semantic_accuracy
    ✅ test de stabilité
```

---

# 9. Choix techniques à préserver

1. **Le code réutilisable reste dans `src/`**, pas dans les notebooks.
2. Les notebooks servent à l'expérimentation et à la démonstration.
3. Utiliser `apply_chat_template()` pour les conversations Qwen.
4. La baseline d'évaluation reste déterministe avec `do_sample=False`.
5. Séparer conformité du format et exactitude métier.
6. Valider les verdicts contre une liste fermée de valeurs autorisées.
7. Garder l'historique borné.
8. Ne pas intégrer définitivement le format du bloc A tant que son contrat n'est pas décidé.
9. Les règles d'âge actuelles sont des **données de test**, pas les règles métier finales.
10. Une réponse incohérente du LLM n'implique pas automatiquement un bug : vérifier séparément le pipeline, le prompt et les capacités du modèle.

---

# 10. Dépendances principales

```text
torch
transformers
accelerate
safetensors
pandas
jupyter
pytest
```

Le modèle est téléchargé via Hugging Face lors du premier chargement.

---

# 11. Ordre conseillé de lecture

```text
1. experiments/tokens_logits.ipynb
2. experiments/decoding.ipynb
3. experiments/baseline.ipynb
4. experiments/prompt_comparison.ipynb
```

Progression :

```text
Transformer
→ décodage
→ agent
→ évaluation
```

---

# 12. Prochaine étape

La prochaine dépendance importante est le **contrat avec le bloc A**.

À clarifier avec l'équipe A :

```text
- structure exacte d'une fiche ;
- structure exacte des règles ;
- format d'échange ;
- champs obligatoires / optionnels ;
- gestion des informations absentes ;
- exemples de fiches réelles ;
- éventuels identifiants ou métadonnées.
```

Une fois ce contrat défini, le bloc B pourra remplacer les fixtures temporaires par les vraies entrées du projet sans modifier le cœur du moteur LLM.
