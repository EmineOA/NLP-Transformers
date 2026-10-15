# Règles spécifiques au bloc D — Évaluation (E6)

Ce fichier complète le `AGENTS.md` à la racine du dépôt. En cas de doute, appliquer aussi les contrats communs de `backlog.md`. Il concerne uniquement le dossier `evaluation/`.

## Mission du bloc D

Mesurer le comportement pédagogique de Brainmaxing de manière reproductible, sans entraîner le modèle et sans confondre les tests du scoreur avec les performances du LLM.

- **T6.1 (Sprint 1)** : préparer cinq scénarios nominaux (N1 à N5) et trois scénarios négatifs (X1 à X3), une grille humaine, un scoreur automatique, puis évaluer la baseline et le mini-LoRA sur les **mêmes** cas.
- **T6.2 (Sprint 2)** : préparer au moins 20 requêtes finales inédites, dont des conversations avec historique et des refus, avant l'entraînement complet.
- **T6.3 (Sprint 3)** : comparer baseline et LoRA sur les mêmes requêtes ; produire des métriques, une évaluation humaine et l'analyse d'au moins cinq réussites ou erreurs.

## Source de vérité et contrats

- Les règles métier et les verdicts attendus doivent être conformes aux fiches et validés avec le bloc A.
- Respecter les contrats de `backlog.md` : verdicts `PARFAIT`, `PARTIEL`, `INCORRECT`, ou `null` pour le hors périmètre ; champs `verdict`, `feedback`, `raw_response`, `valid_format`.
- Ne pas modifier `src/agents/`, `src/llm/`, les fiches pédagogiques ou l'interface Gradio sans coordination avec leurs responsables.
- Le branchement au modèle doit passer par la fonction d'inférence commune définie par l'équipe. Ne pas créer un second moteur LLM propre au scoreur.

## Séparation stricte des données

1. `cases_v0.1.jsonl` contient les requêtes et les **réponses attendues**. Ne jamais inclure les champs `expected` dans les messages transmis au modèle.
2. Les prédictions sont générées **indépendamment**, puis enregistrées dans un fichier distinct pour chaque version de modèle.
3. `demo_score.py` et `test_score.py` utilisent des **prédictions fictives** pour tester le code : leurs scores ne sont jamais présentés comme des performances de Qwen.
4. Les cas réservés à l'évaluation finale ne doivent pas être intégrés au dataset d'entraînement ni utilisés pour optimiser les prompts à répétition.
5. Pour un refus, `verdict: null` et `valid_format: true` ne prouvent pas à eux seuls que le refus est pertinent : le feedback doit être relu et noté séparément.

## Calcul des métriques

- Distinguer scénarios, tours de conversation et tests unitaires : T6.1 comprend **8 scénarios, 9 observations**, dont **5 verdicts nominaux** (N4 contient deux tours).
- X1 vérifie le parseur sur une sortie fabriquée ; ce n'est pas une réponse produite par le modèle.
- Vérifier la conformité du format, l'exactitude des verdicts, les refus et la qualité pédagogique ; ne pas réduire toute l'évaluation à une seule accuracy.
- Noter humainement les cinq critères du sujet sur une échelle 0/1/2 ; utiliser `N/A` quand un critère est inapplicable et le justifier.
- Conserver les versions exactes du modèle, du prompt, des fiches, de l'adaptateur, les paramètres de génération et le commit Git pour toute mesure réelle.

## Vérifications avant commit

Depuis la racine du dépôt :

```bash
python -m unittest discover -s evaluation -p "test_*.py" -v
python evaluation/demo_score.py
```

La démonstration doit rester clairement étiquetée **fictive**. Ne pas annoncer T6.1 comme terminée tant que les sorties réelles de la baseline et du mini-LoRA n'ont pas été mesurées, revues et validées par une autre personne.

## Qualité des contributions

- Changer un seul élément à la fois, expliquer le changement, ajouter son test et faire relire la modification.
- Garder les commentaires et docstrings en français ; noms Python en `snake_case`.
- Privilégier des chemins relatifs et éviter les dépendances GPU dans les tests unitaires du scoreur.
- Ne pas ajouter de nouvelles dépendances ou changer les formats de données sans accord de l'équipe.
- Enregistrer les résultats réels dans des fichiers versionnés distincts (`eval_v0.1_baseline.csv`, `eval_v0.1_lora.csv`) sans écraser les anciennes expériences.
