# Branchement sur le vrai moteur : contrat pour le bloc B

Le fichier `src/agents/baseline.py` du ZIP actuel renvoie `{response, verdict}` avec des verdicts de l'ancien exercice. Pour l'évaluation, le contrat du projet (`backlog.md`) est :

```python
generate_response(messages, concept, question, model_version="baseline")
# => {"verdict": ..., "feedback": ..., "raw_response": ..., "valid_format": ...}
```

Implémentation à coordonner avec Emine, sans faire dépendre le scoreur de `transformers` :

```python
for case in cases:
    if case["type"] == "parser":
        parsed = parse_response(case["raw_response"])
        enregistrer(id=case["id"], tour=1, **parsed)
        # Vérifier séparément que Gradio ne plante pas / ne passe pas à la question suivante.
        continue

    concept = charger_fiche_si_existe(case["concept_id"])
    question = trouver_question(concept, case["question_id"])
    history = []  # Réinitialisé pour chaque scénario.
    for tour, step in enumerate(case["turns"], 1):
        # Ne jamais envoyer step["expected"] au modèle !
        messages = history + [{"role": "user", "content": step["user_input"]}]
        out = generate_response(messages, concept, question, model_version="baseline")
        enregistrer(id=case["id"], tour=tour, **out)
        history = messages + [{"role": "assistant", "content": out["raw_response"]}]
```

Pour X2 (fiche introuvable), l'application doit refuser sans inventer de contenu ; son traitement dépend de la couche de périmètre de l'orchestrateur.

Pour N5 et X3, les demandes sortent des activités autorisées. Le contrat prévoit `verdict=null` et `valid_format=true` pour un refus. Or un parseur seul ne peut pas distinguer un refus d'une sortie mal formée : la couche d'orchestration doit identifier explicitement le refus. Une vérification humaine est ensuite nécessaire pour dire si la réponse a **réellement** refusé.

Format de prédictions JSONL à enregistrer, sans réponses attendues :

```json
{"id":"N2","tour":1,"verdict":"PARTIEL","valid_format":true,"feedback":"Il manque l'explication.","raw_response":"[VERDICT: PARTIEL]\nIl manque l'explication."}
```

Pour un refus, le reviewer peut ajouter `"refus_effectif": true` ou `false` après avoir lu la réponse ; ce champ n'est pas généré automatiquement et ne doit pas être dérivé de `expected`.

Le scoreur vérifie la cohérence entre tags textuels et champs structurés pour les verdicts. Pour le refus sans verdict, le système producteur des prédictions est responsable de `valid_format` (exception du contrat).
