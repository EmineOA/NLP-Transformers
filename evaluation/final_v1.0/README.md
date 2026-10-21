# Évaluation figée T6.2 / T6.3 : 20 requêtes

Suite académique basée sur le format de Jalis / birdyy667 (`dataset_source.csv`, 14 colonnes, séparateur `;`, UTF-8 BOM). La sélection verrouille le blob Git `d2f3f4b8310184705640d8b2afc19a7a66d616c7` provenant du commit `adffd8fc8ac2ac22f6e9620c2b35b10dc08ccf8b`.

- 20 requêtes T01–T20 : 5 `PARFAIT`, 5 `PARTIEL`, 5 `INCORRECT`, 5 `HORS_PERIMETRE`.
- 17 scénarios : 14 indépendants + 3 conversations réelles à 2 tours (C01, C02, C03).
- 3 questions exclusivement dans le split `test` : `bateau-de-thesee_q3`, `doute-de-descartes_q1`, `infinis-plus-grands-que-l-infini_q2`.
- 17 requêtes reprises telles quelles du CSV Jalis, 3 premiers essais reconstruits à partir des historiques de ces réponses, avec feedback de référence créé manuellement.
- Une ligne de CSV par **requête** ; les liens entre requêtes sont décrits dans `selection.json`.
- Aucun `label` ni `feedback` attendu du tour courant n'est envoyé à Qwen. Dans les paires, l'assistant du premier tour est **la sortie réelle du modèle testé**, pas la réponse de référence.

## Mise en place

Depuis la racine du dépôt, une seule fois :

```powershell
python evaluation/run_final.py --prepare
python evaluation/run_final.py --dry-run
python -m unittest discover -s tests -p "test_*.py" -v
```

**Relire les labels et les feedbacks des 20 cas** avant de figer définitivement puis committer `cases.csv`, `lock.json` et `selection.json` ensemble. Les SHA256 des deux premiers fichiers détectent ensuite toute modification accidentelle. Toute révision scientifique du benchmark doit produire `v1.1` et un nouveau manifeste ; il ne faut jamais ajuster v1.0 après avoir vu les scores des modèles.

## Exécution

```powershell
python evaluation/run_final.py --only T11
python evaluation/run_final.py --only T02
python evaluation/run_final.py --model baseline --prompt fewshot
```

`--only T02` rejoue T01 puis T02, car l'historique en dépend. Une exécution partielle enregistre des prédictions, mais **aucun score global**. Sans `--only`, les 20 sorties, le CSV de score et les métriques sont écrits sous `evaluation/final_v1.0/results/`.

Le modèle `lora` est maintenant raccordé à `generate_response` sur `main` (PR #10). Les exécutions enregistrées figurent sous `results/` : baseline 19/20 formats valides, LoRA 20/20 ; verdicts nominaux 11/15 contre 14/15 ; refus formels 0/5 contre 5/5. La notation qualitative LoRA reste une pré-annotation à valider humainement.

La grille humaine existante dans `evaluation/grille_humaine_v0.1.md` est à appliquer aux 20 réponses générées. Conserver un tableau de notation par `test_id` et par `model_version`, avec les cinq critères du sujet et les commentaires. Le taux de refus formel ne suffit pas à valider un refus pertinent.

### Limites méthodologiques

Les 17 entrées existantes proviennent du split `test` de Jalis. Les trois nouvelles réponses `T01`, `T08` et `T14` sont extraites de l'historique de ses réponses à l'essai 2. Elles doivent être relues par le bloc A avant publication. Les questions de test ne doivent jamais être réintroduites dans les données d'entraînement, ni servir à régler le prompt ou les hyperparamètres LoRA.

L'entrée textuelle reprend `build_messages` de Jalis. Pour les trois dialogues, on retire seulement l'essai précédent du **résumé** parce qu'il est transmis comme vrai tour `user/assistant`, tout en conservant les questions antérieures du concept et les indices prévus par `essai_question`.
