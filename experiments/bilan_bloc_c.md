# Bilan du bloc C : adaptation QLoRA du tuteur

Ce document résume le travail QLoRA du bloc C, du chargement des données à l'intégration dans l'application. Tous les chiffres viennent d'exécutions réelles : sorties du notebook [bloc_c_qlora.ipynb](bloc_c_qlora.ipynb) et fichiers de [results/](results/).

## En bref

- Qwen2.5-3B-Instruct, gelé en 4 bits, avec un adaptateur LoRA entraîné sur le dataset du bloc C (885 conversations).
- Sur les 118 exemples de `val` :
  - verdicts justes : **49,2 %** pour le modèle de base, **91,5 %** avec l'adaptateur ;
  - refus (`HORS_PERIMETRE`) détectés : 22 sur 22 avec l'adaptateur, 0 pour la base.
- L'adaptateur final est public sur Hugging Face : [MonsieurNikko/brainmaxing-qlora-v1.0](https://huggingface.co/MonsieurNikko/brainmaxing-qlora-v1.0).
- Il est branché dans l'application : `generate_response(..., model_version="lora")`.

## 1. Ce que le modèle apprend

- **Entrée** : le prompt système (fiche, question, grille de classement), puis le message de l'élève (`RÉPONSE ACTUELLE À ÉVALUER : …`).
- **Sortie apprise** : la réponse du tuteur, c'est-à-dire `[VERDICT: X]` suivi d'une ou deux phrases de feedback.
- **Masque** : le prompt est lu mais pas appris (labels à -100). Seule la réponse du tuteur compte dans la loss. Par exemple, la première conversation a 56 tokens appris et 1 589 masqués.

### Données

| Split | Exemples | PARFAIT | PARTIEL | INCORRECT | HORS_PERIMETRE |
|---|---|---|---|---|---|
| train | 885 | 240 | 240 | 240 | 165 |
| val | 118 | 32 | 32 | 32 | 22 |

- Sources :
  - `data/dialogue/classification/dataset_messages.jsonl` contient les conversations ;
  - `dataset_source.csv` contient la colonne `split`.
- Longueur d'une conversation : de 1 229 à 3 638 tokens, 2 358 en moyenne. D'où `max_length = 4096` : c'est un garde-fou, aucune conversation n'est coupée.
- `val` ne contient que **2 questions** (dilemme du tramway q2, Monty Hall q3).

## 2. Méthode

| Élément | Valeur | Pourquoi |
|---|---|---|
| Modèle de base | Qwen2.5-3B-Instruct, 4 bits nf4 | 1,9 Go sur le GPU au lieu d'environ 6 Go en 16 bits |
| Adaptateur | LoRA r=16, alpha=32, dropout 0,05, 7 types de couches | valeurs courantes, non comparées à d'autres |
| Paramètres entraînés | 29 933 568 sur 3 115 872 256 (0,96 %) | seules les matrices A et B apprennent |
| Lot effectif | 8 conversations (8 x 1 sur A100 80 Go) | loss moyennée par exemple ; 111 pas par epoch |
| Learning rate | 1e-4 | 2e-4 et 1e-4 donnent des loss val proches (un essai chacun) |
| Suivi | loss val tous les 25 pas, patience 4 | arrêt anticipé, meilleur adaptateur gardé dans `best/` |

## 3. Les trois entraînements

| Essai | Matériel | Learning rate | Protocole | Loss val minimale | Meilleur moment |
|---|---|---|---|---|---|
| v0.2 | RTX 4070 (12 Go) | 2e-4 | lot de 1 x 8 accumulations, mesure par epoch | 1.4756 | epoch 1 |
| v0.3 | A100 80 Go (Colab) | 1e-4 | lot de 1 x 8 accumulations, mesure par epoch | 1.4701 | epoch 2 |
| **v1.0 (final)** | A100 80 Go (Colab) | 1e-4 | vrai lot de 8, mesure val tous les 25 pas | **1.4646** | pas 222 (epoch 2) |

- **v0.2** : un exemple prenait environ 18 s sur la RTX 4070, dont le GPU était plein, soit environ 4 h par epoch. C'est ce qui a motivé le passage sur Colab, où un exemple prend environ 1,5 s sur A100.
- **v1.0** : le traitement par vrais lots a été ajouté et vérifié par `--self-test` : les gradients sont égaux à 1e-6 près pour un même lot découpé en 6 x 1, 1 x 6 ou 2 x 3.
  - Arrêt anticipé après 300 pas, en 79 minutes.
  - La loss train passe de 1.96 à 0.83 ; la loss val est minimale (1.465) au pas 222, puis remonte à 1.569. C'est le signe habituel du surapprentissage, ce qui justifie l'arrêt anticipé.
- Les protocoles diffèrent : les trois lignes ne sont **pas comparables ligne à ligne**.

## 4. Résultats

Évaluation sur les mêmes 118 exemples de `val`. Le verdict est lu dans la première ligne de la réponse générée.

| Mesure | Base | v0.3 | **v1.0** |
|---|---|---|---|
| Verdict juste | 49,2 % (58 / 118) | 92,4 % | **91,5 % (108 / 118)** |
| Format valide | 105 / 118 | 118 / 118 | **118 / 118** |
| HORS_PERIMETRE détectés | 0 / 22 | 21 / 22 | **22 / 22** |
| Erreurs graves (PARFAIT donné à une réponse INCORRECT ou HORS_PERIMETRE) | 6 / 54 | – | **0 / 54** |

- v0.3 et v1.0 sont **à égalité** : un exemple d'écart sur 118. v1.0 est retenu parce qu'il détecte tous les refus et qu'il suit le protocole justifié dans le notebook (vrai lot de 8).
- Le rechargement de `best/` (114 Mo) sur Qwen 4 bits a été vérifié : il génère une réponse au bon format, avec le verdict attendu.

### Ce que montrent 5 cas lus à la main

- **Refus** : l'élève demande qu'on rédige à sa place. La base l'évalue comme une réponse normale et donne des éléments de solution ; l'adaptateur refuse et relance l'élève.
- **Feedback** : la base recopie souvent la phrase d'exemple de son prompt (« Tu as identifié un élément… »). L'adaptateur parle de ce que l'élève a réellement écrit.
- **Erreurs de l'adaptateur** :
  - un verdict trop indulgent (PARTIEL au lieu d'INCORRECT) quand l'élève conteste une donnée de l'énoncé ;
  - un feedback faux malgré un verdict juste. La mesure ne note que le verdict.

## 5. Intégration dans l'application

- **Code** : `src/agents/inference.py`. `generate_response(..., model_version="lora")` charge Qwen en 4 bits, puis branche l'adaptateur de `adapters/v1.0/best`. Le chargement n'a lieu qu'une fois.
  - Le prompt, l'historique et la lecture du verdict sont identiques à ceux de la baseline. Le contrat 3 est inchangé.
- **Tests** : `tests/test_bloc_b.py` contient 14 tests, tous OK, avec un faux modèle et sans GPU. Ils vérifient que :
  - `"lora"` passe par l'adaptateur ;
  - une version inconnue est refusée.
- **Adaptateur** : il est public sur Hugging Face, donc on peut l'utiliser sans compte et sans réentraîner :

```python
from huggingface_hub import snapshot_download
snapshot_download("MonsieurNikko/brainmaxing-qlora-v1.0", local_dir="adapters/v1.0/best")

from src.agents.inference import generate_response
result = generate_response(messages, "monty-hall", "monty-hall_q1", model_version="lora")
```

- Vérifié sur Colab : l'adaptateur est téléchargé depuis Hugging Face, puis `generate_response` renvoie `PARFAIT | Tu utilises le savoir du présentateur…`.

## 6. Limites

- **`val` est étroit et a servi à choisir l'adaptateur.** Il ne contient que 2 questions, avec des exemples très proches entre eux, et c'est lui qui a désigné le meilleur pas. Les scores sont donc optimistes. Le test indépendant est celui du bloc D (évaluation).
- **Un seul run par réglage.** Le rang, l'alpha, le dropout, les couches et le nombre d'epochs n'ont pas été comparés à d'autres valeurs.
- **Comparaison base contre LoRA.** Dans l'évaluation du bloc C, les deux modèles sont en 4 bits. Dans l'application, la baseline est chargée en 16 bits. Une comparaison faite dans l'application mélange donc l'effet de l'adaptateur et celui de la quantification.
- **Format du message élève.** À l'entraînement, il commençait toujours par `RÉPONSE ACTUELLE À ÉVALUER :`. Si l'interface envoie le texte brut, le modèle reçoit un format qu'il n'a jamais vu. C'est à décider avec le bloc B.
- **Petit GPU.** La génération sur un GPU plus petit que l'A100 (par exemple une T4) n'a pas encore été essayée.

## 7. Reste à faire

- Décider avec Emine (bloc B) qui ajoute le préfixe `RÉPONSE ACTUELLE À ÉVALUER :`.
- Ajouter le lien Hugging Face au README.
- Copier les résultats v1.0 (CSV, `config.json`, `resultat.json`, évaluations) dans `experiments/results/train_v1.0` et `results/eval/`.
- Faire relire la branche `bloc-c-entrainement` par une autre personne, puis la fusionner sur `main`.
- Brancher le LoRA dans Gradio (T7.1) et dans l'évaluation du bloc D.

## Fichiers

| Fichier | Rôle |
|---|---|
| `src/training/dataset.py` | chargement des splits, tokenisation, masque du prompt, lots |
| `src/training/model.py` | Qwen en 4 bits, ajout et rechargement de l'adaptateur |
| `src/training/train.py` | loss par exemple, pas d'optimisation, loss val, génération |
| `src/training/run_training.py` | entraînement complet : arrêt anticipé, `config.json`, `resultat.json` |
| `src/training/evaluate_verdicts.py` | exactitude du verdict, base ou adaptateur |
| `src/training/export_examples.py` | cas base contre LoRA, équilibrés entre les questions |
| `src/training/plot_losses.py` | courbes de loss |
| `src/agents/inference.py` | `generate_response`, avec `model_version="lora"` |
| `experiments/bloc_c_qlora.ipynb` | notebook du bloc C, à lire avec ses sorties |
| `requirements-qlora.txt` | versions figées (transformers 5.20.1, peft 0.21.2, bitsandbytes 0.50.2, accelerate 1.15.0) |
