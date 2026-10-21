# Bilan du bloc C : adaptation QLoRA du tuteur

Ce document résume le travail QLoRA du bloc C, du chargement des données à l'intégration dans l'application. Tous les chiffres viennent d'exécutions réelles : sorties du notebook [bloc_c_qlora.ipynb](bloc_c_qlora.ipynb) et fichiers de [results/](results/).

## En bref

- Qwen2.5-3B-Instruct, gelé en 4 bits, avec un adaptateur LoRA entraîné sur le dataset du bloc A (1 003 conversations).
- **Nouveau split** : train / val / test = 649 / 177 / 177, découpé par question entière (11 / 3 / 3 questions). Aucune question n'est partagée entre deux splits. `test` n'a servi ni à l'entraînement ni au choix de l'adaptateur.
- **Run v1.1** (sur ce split), sur les 177 exemples de `test` (questions jamais vues) :
  - verdicts justes : **52,5 %** pour le modèle de base, **86,4 %** avec l'adaptateur (85,9 % sur `val`) ;
  - refus (`HORS_PERIMETRE`) détectés : 31 sur 33 avec l'adaptateur, 0 pour la base ;
  - format valide 177 / 177 et aucune erreur grave (0 / 81) avec l'adaptateur.
- **Run v1.0** (ancien split 885 / 118, sans test) : sur les 118 exemples de `val`, verdicts justes **49,2 %** pour la base et **91,5 %** avec l'adaptateur ; refus détectés 22 / 22. Ces chiffres ne sont pas comparables à ceux de v1.1 : les questions du nouveau `val` et du nouveau `test` faisaient partie de l'ancien `train`.
- L'adaptateur v1.1 est public sur Hugging Face : [MonsieurNikko/brainmaxing-qlora-v1.1](https://huggingface.co/MonsieurNikko/brainmaxing-qlora-v1.1). Il est branché dans l'application : `generate_response(..., model_version="lora")`. L'adaptateur v1.0 (ancien split) reste public : [MonsieurNikko/brainmaxing-qlora-v1.0](https://huggingface.co/MonsieurNikko/brainmaxing-qlora-v1.0).

## 1. Ce que le modèle apprend

- **Entrée** : le prompt système (rôle du tuteur, fiche, question actuelle, grille de classement et règles), puis le message de l'élève. Ce message est construit par `data/dialogue/classification/export_jsonl.py` en jusqu'à trois blocs, séparés par une ligne vide :
  1. `HISTORIQUE RÉSUMÉ (contexte, pas la réponse à évaluer) :` puis les essais précédents et leur verdict (829 conversations sur 1 003) ;
  2. `INDICES DÉJÀ FOURNIS (selon le numéro d'essai) :` puis les indices de la fiche déjà donnés (510 sur 1 003, à partir du 2e essai) ;
  3. `RÉPONSE ACTUELLE À ÉVALUER :` puis la réponse de l'élève (toujours présent, 1 003 sur 1 003).
- **Sortie apprise** : la réponse du tuteur, c'est-à-dire `[VERDICT: X]` suivi d'une ou deux phrases de feedback.
- **Masque** : le prompt est lu mais pas appris (labels à -100). Seule la réponse du tuteur compte dans la loss. Par exemple, la première conversation a 56 tokens appris et 1 589 masqués.

### Données

| Split | Questions | Exemples | PARFAIT | PARTIEL | INCORRECT | HORS_PERIMETRE |
|---|---|---|---|---|---|---|
| train | 11 | 649 | 176 | 176 | 176 | 121 |
| val | 3 | 177 | 48 | 48 | 48 | 33 |
| test | 3 | 177 | 48 | 48 | 48 | 33 |

- **Découpage par question entière** (bloc A, commit `c46fece`) : toutes les réponses à une même question sont dans le même split. Le modèle est donc mesuré sur des questions qu'il n'a jamais vues, pas sur des reformulations de réponses vues à l'entraînement. `python -m src.training.dataset` vérifie qu'aucune question n'est partagée.
  - `val` : dilemme du tramway q2, Monty Hall q3, plier une feuille 42 fois q2 ;
  - `test` : bateau de Thésée q3, doute de Descartes q1, infinis plus grands que l'infini q2.
- **Rôle des splits** : `train` pour apprendre, `val` pour suivre la loss et choisir l'adaptateur retenu (`best/`), `test` pour la mesure finale uniquement.
- Sources :
  - `data/dialogue/classification/dataset_messages.jsonl` contient les conversations ;
  - `dataset_source.csv` contient la colonne `split` (la ligne i du CSV correspond à la ligne i du JSONL).
- Longueur d'une conversation : de 1 229 à 3 638 tokens, 2 358 en moyenne. D'où `max_length = 4096` : c'est un garde-fou, aucune conversation n'est coupée.
- **Ancien split** (runs v0.2 à v1.0) : 885 train / 118 val, sans test ; `val` ne contenait que 2 questions (dilemme du tramway q2, Monty Hall q3).

## 2. Méthode

| Élément | Valeur | Pourquoi |
|---|---|---|
| Modèle de base | Qwen2.5-3B-Instruct, 4 bits nf4 | 1,9 Go sur le GPU au lieu d'environ 6 Go en 16 bits |
| Adaptateur | LoRA r=16, alpha=32, dropout 0,05, 7 types de couches | valeurs courantes, non comparées à d'autres |
| Paramètres entraînés | 29 933 568 sur 3 115 872 256 (0,96 %) | seules les matrices A et B apprennent |
| Lot effectif | 8 conversations : 8 x 1 sur un GPU de plus de 60 Go (A100 80 Go), sinon 4 x 2 (L4, T4) | loss moyennée par exemple ; même lot effectif quelle que soit la découpe. 111 pas par epoch sur l'ancien train (885) |
| Learning rate | 1e-4 | 2e-4 et 1e-4 donnent des loss val proches (un essai chacun) |
| Suivi | loss val tous les 25 pas, patience 4 | arrêt anticipé, meilleur adaptateur gardé dans `best/` |

## 3. Les entraînements

| Essai | Split | Matériel | Learning rate | Protocole | Loss val minimale | Meilleur moment |
|---|---|---|---|---|---|---|
| v0.2 | ancien (885 / 118) | RTX 4070 (12 Go) | 2e-4 | lot de 1 x 8 accumulations, mesure par epoch | 1.4756 | epoch 1 |
| v0.3 | ancien (885 / 118) | A100 80 Go (Colab) | 1e-4 | lot de 1 x 8 accumulations, mesure par epoch | 1.4701 | epoch 2 |
| v1.0 | ancien (885 / 118) | A100 80 Go (Colab) | 1e-4 | vrai lot de 8, mesure val tous les 25 pas | 1.4646 | pas 222 (epoch 2) |
| **v1.1 (final)** | **nouveau (649 / 177 / 177)** | A100 80 Go (Colab) | 1e-4 | vrai lot de 8, mesure val tous les 25 pas | **1.4659** | pas 164 (epoch 2) |

- v1.1 reprend exactement les réglages de v1.0 ; seul le split change. Sa loss val n'est pas comparable à celles des runs précédents : ce n'est pas le même `val`.
- **v1.1** : 82 pas par epoch (649 exemples, lot de 8). Arrêt anticipé après 246 pas, en 70 minutes ; mémoire GPU maximale 44,12 Go.
  - La loss train passe de 1.97 à 0.88 (moyennes des 10 premiers et 10 derniers pas) ; la loss val est minimale (1.466) au pas 164, puis remonte à 1.562. Même signe de surapprentissage qu'en v1.0, d'où l'arrêt anticipé.

- **v0.2** : un exemple prenait environ 18 s sur la RTX 4070, dont le GPU était plein, soit environ 4 h par epoch. C'est ce qui a motivé le passage sur Colab, où un exemple prend environ 1,5 s sur A100.
- **v1.0** : le traitement par vrais lots a été ajouté et vérifié par `--self-test` : les gradients sont égaux à 1e-6 près pour un même lot découpé en 6 x 1, 1 x 6 ou 2 x 3.
  - Arrêt anticipé après 300 pas, en 79 minutes.
  - La loss train passe de 1.96 à 0.83 ; la loss val est minimale (1.465) au pas 222, puis remonte à 1.569. C'est le signe habituel du surapprentissage, ce qui justifie l'arrêt anticipé.
- Les protocoles diffèrent : les trois lignes ne sont **pas comparables ligne à ligne**.

## 4. Résultats

Le verdict est lu dans la première ligne de la réponse générée. Génération gloutonne : la mesure est reproductible.

### Run v1.1 (nouveau split)

Sorties des cellules d'évaluation du notebook (fichiers `eval_v1.1_*.json` sur le Drive).

| Mesure | `val` base | `val` v1.1 | **`test` base** | **`test` v1.1** |
|---|---|---|---|---|
| Verdict juste | 53,7 % (95 / 177) | 85,9 % (152 / 177) | 52,5 % (93 / 177) | **86,4 % (153 / 177)** |
| Format valide | 158 / 177 | 177 / 177 | 162 / 177 | **177 / 177** |
| HORS_PERIMETRE détectés | 0 / 33 | 31 / 33 | 0 / 33 | **31 / 33** |
| Erreurs graves (PARFAIT donné à une réponse INCORRECT ou HORS_PERIMETRE) | 10 / 81 | 0 / 81 | 7 / 81 | **0 / 81** |

- `val` a servi à choisir l'adaptateur : ses chiffres sont optimistes. Le chiffre à retenir est celui de `test`. Les deux sont proches (85,9 % et 86,4 %) : l'adaptateur généralise à des questions jamais vues.
- **Erreur la plus fréquente de l'adaptateur** : trop sévère sur PARTIEL. Sur `test`, 14 des 48 PARTIEL sont notés INCORRECT (12 sur `val`). Il ne donne jamais PARFAIT à une réponse INCORRECT ou hors périmètre.
- **Modèle de base** : il ne refuse jamais (0 / 33) et produit des formats invalides (15 sur `test`), par exemple une balise `[VERDICT: HORS_PERIMETRE]` qui n'existe pas.

### Ce que montrent 5 cas lus à la main (run v1.1, `val`)

- **Feedback** : la base recopie la phrase d'exemple de son prompt (« Tu as identifié un élément. Peux-tu préciser ton raisonnement ? »). L'adaptateur reprend ce que l'élève a réellement écrit.
- **Refus** : l'élève demande la hauteur au 41ᵉ pli au lieu de la trouver. La base note PARFAIT et donne la réponse ; l'adaptateur refuse et relance l'élève à partir de son propre raisonnement.
- **Indice inventé** : l'élève choisit une porte pour sa « poignée plus neuve ». L'adaptateur donne INCORRECT et ramène l'élève à ce que sait le présentateur ; la base produit une balise invalide.
- **Erreur de l'adaptateur** : PARFAIT au lieu de PARTIEL sur Monty Hall q3, avec un feedback vague (« Ton calcul suit ce raisonnement ») qui ne vérifie pas le calcul. La base a le bon verdict, mais son feedback affirme 1/3 contre 2/3 au lieu de 1/2 – 1/2. La mesure ne note que le verdict : la cohérence entre verdict et feedback n'est pas évaluée.

### Runs précédents (ancien split, 118 exemples de `val`)

| Mesure | Base | v0.3 | **v1.0** |
|---|---|---|---|
| Verdict juste | 49,2 % (58 / 118) | 92,4 % | **91,5 % (108 / 118)** |
| Format valide | 105 / 118 | 118 / 118 | **118 / 118** |
| HORS_PERIMETRE détectés | 0 / 22 | 21 / 22 | **22 / 22** |
| Erreurs graves (PARFAIT donné à une réponse INCORRECT ou HORS_PERIMETRE) | 6 / 54 | – | **0 / 54** |

- v0.3 et v1.0 sont **à égalité** : un exemple d'écart sur 118. v1.0 est retenu parce qu'il détecte tous les refus et qu'il suit le protocole justifié dans le notebook (vrai lot de 8).
- Le rechargement de `best/` (114 Mo) sur Qwen 4 bits a été vérifié : il génère une réponse au bon format, avec le verdict attendu.

### Ce que montrent 5 cas lus à la main (run v1.0)

- **Refus** : l'élève demande qu'on rédige à sa place. La base l'évalue comme une réponse normale et donne des éléments de solution ; l'adaptateur refuse et relance l'élève.
- **Feedback** : la base recopie souvent la phrase d'exemple de son prompt (« Tu as identifié un élément… »). L'adaptateur parle de ce que l'élève a réellement écrit.
- **Erreurs de l'adaptateur** :
  - un verdict trop indulgent (PARTIEL au lieu d'INCORRECT) quand l'élève conteste une donnée de l'énoncé ;
  - un feedback faux malgré un verdict juste. La mesure ne note que le verdict.

## 5. Intégration dans l'application

- **Code** : `src/agents/inference.py`. `generate_response(..., model_version="lora")` charge Qwen en 4 bits, puis branche l'adaptateur de `adapters/v1.1/best`. Le chargement n'a lieu qu'une fois.
  - Le prompt, l'historique et la lecture du verdict sont identiques à ceux de la baseline. Le contrat 3 est inchangé.
- **Tests** : `tests/test_bloc_b.py` contient 14 tests, tous OK, avec un faux modèle et sans GPU. Ils vérifient que :
  - `"lora"` passe par l'adaptateur ;
  - une version inconnue est refusée.
- **Adaptateur** : il est public sur Hugging Face, donc on peut l'utiliser sans compte et sans réentraîner :

```python
from huggingface_hub import snapshot_download
snapshot_download("MonsieurNikko/brainmaxing-qlora-v1.1", local_dir="adapters/v1.1/best")

from src.agents.inference import generate_response
result = generate_response(messages, "monty-hall", "monty-hall_q1", model_version="lora")
```

- Vérifié sur Colab : l'adaptateur est téléchargé depuis Hugging Face, puis `generate_response` renvoie `PARFAIT | Tu utilises le savoir du présentateur…`.

## 6. Limites

- **`val` et `test` sont petits.** 3 questions chacun, avec plusieurs exemples très proches par question. `val` a en plus désigné le meilleur pas : ses scores sont optimistes. `test` (3 questions jamais vues) donne le chiffre à retenir, mais une seule question difficile peut le faire varier beaucoup.
- **Un seul run par réglage.** Le rang, l'alpha, le dropout, les couches et le nombre d'epochs n'ont pas été comparés à d'autres valeurs.
- **Comparaison base contre LoRA.** Dans l'évaluation du bloc C, les deux modèles sont en 4 bits. Dans l'application, la baseline est chargée en 16 bits. Une comparaison faite dans l'application mélange donc l'effet de l'adaptateur et celui de la quantification.
- **Format du message élève.** À l'entraînement, le message de l'élève suit toujours le format de la section 1 : historique résumé et indices éventuels, puis `RÉPONSE ACTUELLE À ÉVALUER :` suivi de la réponse. Si l'interface envoie le texte brut, le modèle reçoit un format qu'il n'a jamais vu. C'est à décider avec le bloc B.
- **Petit GPU.** La génération sur un GPU plus petit que l'A100 (par exemple une T4) n'a pas encore été essayée.

## 7. Reste à faire

- Mesurer la cohérence entre le verdict et le feedback (cas 4 de la section 4).
- Harmonisation du format élève proposée dans la PR #12 : sérialiseur partagé entre export et Gradio, à relire avec B et E.
- Intégration à `main` effectuée via PR #10 ; vérifier le fonctionnement dans le notebook académique final.
- L'évaluation du bloc D sur LoRA est enregistrée ; vérifier l'interface Gradio sur GPU réel après fusion de la PR d'intégration.

## Fichiers

| Fichier | Rôle |
|---|---|
| `src/training/dataset.py` | chargement des splits train / val / test, tokenisation, masque du prompt, lots |
| `src/training/model.py` | Qwen en 4 bits, ajout et rechargement de l'adaptateur |
| `src/training/train.py` | loss par exemple, pas d'optimisation, loss val, génération |
| `src/training/run_training.py` | entraînement complet : arrêt anticipé, `config.json`, `resultat.json` |
| `src/training/evaluate_verdicts.py` | exactitude du verdict, base ou adaptateur, sur `val` ou `test` (`--split`) |
| `src/training/export_examples.py` | cas base contre LoRA, équilibrés entre les questions |
| `src/training/plot_losses.py` | courbes de loss |
| `src/agents/inference.py` | `generate_response`, avec `model_version="lora"` |
| `experiments/bloc_c_qlora.ipynb` | notebook du bloc C, à lire avec ses sorties |
| `experiments/results/train_v1.1/`, `results/eval/*_v1.1*` | historiques de loss, config et évaluations (`val` et `test`) du run final v1.1 |
| `experiments/results/train_v1.0/`, `results/eval/*_v1.0*` | historiques de loss, config et évaluations du run v1.0 (ancien split) |
| `requirements-qlora.txt` | versions figées (transformers 5.20.1, peft 0.21.2, bitsandbytes 0.50.2, accelerate 1.15.0) |
