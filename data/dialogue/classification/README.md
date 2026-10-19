# Dataset de classification : méthode de génération et de validation

Ce dossier contient la source du dataset d'entraînement QLoRA du tuteur Brainmaxing. Ce README décrit **comment on le génère et comment on vérifie qu'il est cohérent**. Le format des colonnes et les règles de l'historique sont décrits dans [`AGENTS.md`](AGENTS.md).

## Livraison et utilisation par Emine

Les fichiers à utiliser sont :

| Fichier | Usage |
|---|---|
| `dataset_source.csv` | Source unique à corriger : 1 003 lignes, 14 colonnes, identifiants, split et générateur conservés. |
| `dataset_messages.jsonl` | Export commun Opus + Sol : 1 003 objets au format `{"messages": [...]}`. |
| `export_jsonl.py` | Reproduit l'export depuis le CSV et les fiches, sans générer de texte. |
| `generation/AGENTS.md` | Méthode, répartition et liste exhaustive des 472 identifiants Opus et 531 identifiants Sol. |
| `generation/manifest.json` | Versions, corrections, empreintes et provenance des sorties archivées. |

Depuis la racine du dépôt :

```bash
python data/dialogue/classification/valider_dataset.py
python data/dialogue/classification/export_jsonl.py
python -m unittest discover -s tests -p 'test_*.py' -v
```

L'exporteur n'a pas de dépendance supplémentaire : il utilise la bibliothèque standard et les fonctions existantes du dépôt. La suite complète de tests nécessite les dépendances du moteur, dont PyTorch et Transformers.

Chaque ligne JSONL contient seulement `messages`, avec trois messages :

1. `system` : prompt produit par `src/llm/prompts.py`, version `fewshot` par défaut, avec la fiche et la question réelles. Les exemples de fiche sont ceux du moteur, pas les cibles du CSV.
2. `user` : historique résumé conservé en texte, indices déduits du numéro d'essai, puis réponse actuelle de l'élève. Aucun ancien feedback complet n'est inventé.
3. `assistant` : `[VERDICT: PARFAIT|PARTIEL|INCORRECT]` ou `[HORS_PERIMETRE]`, puis un saut de ligne et le feedback cible.

Le JSONL garde l'ordre du CSV. L'objet JSONL à l'indice `i` correspond à la ligne CSV à l'indice `i` après lecture avec `csv.DictReader`. `id`, `cas`, `style`, `labels_historique`, `split` et `generateur` ne sont pas injectés dans les messages. Les identifiants contiennent le label : ne pas les ajouter au prompt lors de l'évaluation.

Pour l'entraînement, conserver les trois messages et calculer la perte sur la cible `assistant`. Pour la baseline ou l'évaluation, séparer les entrées et la cible :

```python
model_inputs = example["messages"][:-1]
expected_output = example["messages"][-1]["content"]
```

`generate_response` reconstruit lui-même le message système et refuse les messages système externes. Pour utiliser ce moteur avec la ligne CSV correspondante, lui passer `example["messages"][1:-1]`, la fiche chargée par `concept_id`, la question chargée par `question_id` et `prompt_version="fewshot"`. Ne jamais lui envoyer le dernier message `assistant` du JSONL. Le test `tests/test_dataset_integration.py` vérifie sur les 1 003 lignes que les messages finalement transmis au backend sont exactement ceux de l'export, cible exclue.

Le résumé est une représentation explicite de l'historique disponible, pas une reconstitution d'anciens tours complets. Le moteur reste inchangé. Pour une comparaison baseline/LoRA équitable, les deux doivent recevoir cette même représentation ; une application qui envoie de vrais échanges complets doit être alignée avec l'équipe avant de revendiquer cette équivalence.

Le split est dans le CSV, groupé par question : 649 lignes `train` (11 questions), 177 lignes `val` (3 questions) et 177 lignes `test` (3 questions). Le JSONL commun contient les trois : appliquer la colonne `split` avant l'entraînement, sans utiliser tout l'export comme un unique lot train.

**État des contrôles techniques :** 1 003 lignes `OK` dans le validateur, 1 003 cibles acceptées par le parseur, 18 tests passés avec les dépendances du moteur. L'intégration utilise un backend simulé : aucun score ni entraînement Qwen n'est mesuré ici. Ces contrôles ne valent pas validation pédagogique indépendante ; le juge et l'audit humain restent à faire.

Les deux dossiers de sorties JSON redondantes ont été retirés après comparaison exacte avec le CSV et sauvegarde. Leurs contenus et chemins restent récupérables au commit `4e60507`. Les trois anciens fichiers `bateau-de-thesee-q1.json`, `-q2.json` et `-q3.json` ont aussi été retirés ; ils restent récupérables au commit `000f3f7`. Seul `premier_test.json` reste comme archive pilote dans ce dossier, hors export. Corriger le CSV, relancer les contrôles, puis régénérer le JSONL ; ne pas modifier ce dernier à la main.

## 1. Objectif

Produire un dataset de **1 003 lignes**. Chaque ligne contient une réponse d'élève, le verdict attendu (`PARFAIT`, `PARTIEL`, `INCORRECT` ou `HORS_PERIMETRE`) et le feedback du tuteur. Il faut aussi **prouver que ces lignes sont justes** avant de les utiliser pour entraîner le modèle.

## 2. Pourquoi 1 003 lignes

Le backlog prévoyait environ 300 dialogues. Ce n'est pas assez pour couvrir la grille des fiches.

Pour une seule question, le tuteur doit gérer 4 verdicts × 3 essais × 3 styles d'élève, soit **36 situations**. À cela s'ajoutent les critères de la fiche : 2 à 3 cas `PARTIEL`, 2 à 3 cas `INCORRECT` et 4 types de refus (`HORS_PERIMETRE`).

| | ≈ 300 lignes | 1 003 lignes |
|---|---|---|
| Lignes par question | ≈ 18 | 59 |
| Exemples par critère `PARTIEL` / `INCORRECT`, par question | 1 à 2 | 5 à 8 |
| Exemples par type de refus (`HORS_PERIMETRE`), par question | moins de 1 | 2 à 3 (≈ 47 sur le corpus) |
| Lignes par (question, verdict, essai) | souvent 0 ou 1 | au moins 3 |
| Lignes de validation / test | ≈ 35 / ≈ 35 | 177 / 177 |

Avec 300 lignes, certains critères des fiches n'apparaîtraient jamais à l'entraînement. Avec 1 003 lignes, chaque critère est illustré plusieurs fois, dans des essais et des styles différents.

L'ordre de grandeur correspond à la littérature :
- **LIMA** (Zhou et al., 2023) : environ 1 000 exemples soignés suffisent pour apprendre un format et un style de réponse ;
- **QLoRA** (Dettmers et al., 2023) : la qualité des données compte davantage que leur quantité.

Enfin, une taille plus grande ne coûte pas plus cher à vérifier. La précision de l'audit humain dépend du nombre de lignes relues (section 5), pas de la taille du dataset.

**Limite assumée :** les refus (`HORS_PERIMETRE`) sont moins représentés par question. Comme un refus a la même forme quelle que soit la question, la couverture se juge plutôt sur l'ensemble du corpus.

**Vérification possible :** entraîner le LoRA sur 300, 600 et 1 003 lignes et comparer la perte de validation. Si le modèle progresse encore au-delà de 600 lignes, 300 n'aurait pas suffi.

## 3. Ce qui a déjà été fait

**Source fusionnée (16/10/2026).** `dataset_source.csv` contient les 1 003 lignes avec `label`, `cas`, `style`, `essai_question`, `labels_historique` et `split` fixés par le squelette, puis `generateur` attribué aléatoirement (section 4). Les colonnes `historique`, `reponse_utilisateur` et `feedback` sont maintenant remplies à partir des 34 fichiers de `generation/claude-opus-5.5/` et `generation/sol-6.1/`, par jointure exacte sur `id`. L'ordre, les labels et toutes les autres colonnes sont conservés. Les 174 historiques vides correspondent uniquement à Q1, essai 1. Les labels sont **imposés pour la génération, pas encore validés indépendamment** : les contrôles automatiques ne remplacent ni le juge ni l'audit humain des étapes 4 et 5. La traçabilité de la fusion figure dans `generation/manifest.json`.

**Attention :** `generer_squelette.py` puis `attribuer_generateurs.py` reproduisent le squelette vide et l'attribution, pas les textes générés. Ne pas relancer la génération du squelette sur ce CSV rempli sans en conserver une copie.

**Premier test (`premier_test.json`, 24 lignes de Bateau de Thésée Q1).**
1. **Version 1** (commit `ac08b75`) : 14 lignes générées par Anthusan avec Claude Opus 5.5 et la première version du prompt.
2. **Relecture** de ces 14 lignes. Les feedbacks étaient corrects sur le fond, mais trop scolaires et trop « IA » :
   - ils commençaient par reformuler la réponse de l'élève (« Ta règle est nette… ») ;
   - ils utilisaient des tics comme « en découle » ou « montres clairement » ;
   - ils étaient tous construits sur le même modèle et ne s'appuyaient pas sur l'erreur de l'élève pour le faire avancer.
3. **Correction du prompt** (commit `73962d6`, section 5 de `prompt_a_donner.md` réécrite) :
   - le tuteur commence par **réagir** à ce que l'élève a écrit, au lieu de reformuler sa réponse ;
   - il **vulgarise** avec des images du quotidien et se sert de l'erreur pour montrer pourquoi le raisonnement coince, sans donner la réponse ;
   - il s'adapte au ton de l'élève mais **reste le professeur** ;
   - la longueur est libre, et les règles par verdict sont précisées : pas de question en `PARFAIT`, une question qui fait avancer en `PARTIEL` et `INCORRECT` ;
   - une passe **Humanizer** (section 5.4) traque 10 tics d'écriture d'IA.
4. **Version 2** : 10 lignes générées avec le nouveau prompt (`parfait_15`, `parfait_16`, `partiel_01` à `partiel_08`), à comparer aux 14 premières.

**Ancienne génération Bateau de Thésée Q1 à Q3** (`bateau-de-thesee-q1.json`, `-q2.json`, `-q3.json`) : 59 lignes par question, conservées dans l'historique Git au commit `000f3f7`. Ces fichiers ont été retirés du dossier de travail ; ils ne sont pas une source de l'export actuel.

**Décision (16/10) : tout regénérer.** Les 177 anciennes lignes du Bateau de Thésée restent accessibles dans Git. Les 24 lignes de `premier_test.json` restent dans le dossier comme trace. Aucune de ces archives ne fait partie du dataset final. Les 1 003 lignes ont été régénérées avec le protocole ci-dessous, pour que tout le dataset suive le même prompt, la même attribution des générateurs et les mêmes contrôles.

**Convention du refus (16/10).** On s'aligne sur le moteur (`src/agents/baseline.py`, `src/llm/prompts.py`) : le label `REFUS` devient `HORS_PERIMETRE`, et le modèle écrit `[HORS_PERIMETRE]` en première ligne à la place de `[VERDICT: ...]`, suivi d'un feedback qui recadre. Le changement est appliqué au CSV (`label`, `labels_historique`, `id`), aux JSON générés, au prompt et à `AGENTS.md`. Dans les fiches concept, le niveau garde son nom `refus`, car le chargeur `src/agents/concepts.py` l'exige.

## 4. La méthode

Le générateur reçoit le label imposé et écrit une réponse censée le mériter. Rien ne garantit qu'elle le mérite vraiment. Toute la méthode sert donc à **vérifier le dataset par d'autres moyens que le générateur lui-même**.

### Étape 1 : figer les versions

Avant de générer, on note dans `generation/manifest.json` :
- la version des fiches ;
- la version du prompt ;
- le modèle générateur et ses paramètres ;
- la date.

**Deux générateurs, attribués au hasard.** Les lignes sont générées depuis l'agent Hermes, avec un effort de raisonnement moyen pour les deux modèles :

| Générateur | Lignes |
|---|---|
| Claude Opus 5.5 (`claude-opus-5.5`) | 472 |
| Sol 6.1 (`sol-6.1`) | 531 |

`attribuer_generateurs.py` fait un tirage sans remise (graine 2026) : il tire un indice au hasard parmi les x lignes restantes, puis parmi les x − 1 suivantes, etc. Les 472 lignes tirées vont à Opus, les autres à Sol. Chaque question a entre 23 et 35 lignes Opus.

Avoir deux générateurs permet de vérifier que les résultats ne dépendent pas d'un seul modèle : l'audit (section 5) compare l'accord humain sur les lignes de chacun.

**Remarque :** le tirage n'est pas stratifié, donc la répartition des verdicts n'est pas identique entre les deux modèles (par exemple 114 `PARFAIT` pour Opus contre 158 pour Sol). Ça n'empêche pas la comparaison, mais il faut la faire verdict par verdict.

**Où allaient les sorties brutes :** `generation/<generateur>/<question_id>.json`, au format exact du prompt. Ces 34 fichiers ont été fusionnés dans le CSV puis retirés ; leur provenance et leur liste d'identifiants restent dans `generation/AGENTS.md`. `preparer_lot.py` permet encore de préparer un lot de correction pour une question et un générateur.

### Étape 2 : pilote

On génère une question complète, on la relit et on corrige le prompt si besoin. Les autres questions ne sont générées qu'**une fois le prompt gelé**. Toute modification ultérieure du prompt impose de regénérer les questions déjà produites.

### Étape 3 : contrôles automatiques (script, sans LLM)

Script : `valider_dataset.py`. Il écrit `generation/rapport_validation.csv` avec, pour chaque ligne, `OK`, `ALERTE` (à relire) ou `REJET` (à regénérer).

- **Structure** : `id` conservés, aucun `;`, aucun `[VERDICT` dans le feedback, JSON valide.
- **Historique** : il respecte `labels_historique` (nombre d'éléments, ordre, verdicts, format exact).
- **Feedback selon le verdict** :
  - `PARFAIT` : aucune question ;
  - `PARTIEL` et `INCORRECT` : se termine par une question ;
  - `HORS_PERIMETRE` : relance la question.
- **Fuite de la réponse** : pas de liste de termes interdits. Elle est vérifiée par le LLM juge (étape 4) et par l'audit humain (« le feedback donne-t-il la réponse ? »).
- **Tics d'IA** : recherche des formules listées dans la section 5 du prompt.
- **Diversité** : repérer les quasi-doublons entre lignes et les copies des exemples des fiches, et regarder la répartition des longueurs.

### Étape 4 : LLM juge indépendant

Un modèle d'**une autre famille que les deux générateurs** (par exemple Gemini) reçoit la fiche et la réponse de l'élève, **sans le label**, et propose un verdict.
- En cas d'accord avec le label imposé, la ligne est acceptée.
- En cas de désaccord, la ligne part en relecture humaine.

Le juge note aussi le feedback selon la grille de la section 5. On n'utilise jamais Qwen comme juge, puisque c'est le modèle évalué.

### Étape 5 : audit humain

Détaillé à la section 5.

### Étape 6 : correction

On regénère **uniquement** les lignes rejetées, avec au maximum 2 nouvelles tentatives par ligne. On note les taux de rejet à chaque étape, pour le rapport.

### Étape 7 : séparer entraînement et évaluation

On vérifie qu'aucune ligne du dataset n'est une quasi-copie des cas de test du bloc D (`evaluation/cases_v0.1.jsonl`, puis les 20 tests de T6.2) ni des cas de développement du bloc B (`experiments/dev_cases_bloc_b.jsonl`).

## 5. Audit humain et accord inter-annotateurs

### Protocole

- **À l'aveugle** : l'annotateur voit l'énoncé, l'historique et la réponse de l'élève, mais **ni le label imposé ni le `cas`**. Sinon il aurait tendance à confirmer le label.
- **Tirage aléatoire stratifié** : environ **100 lignes**, avec au moins 20 lignes par verdict et toutes les questions et tous les essais représentés. Avec 100 lignes, le taux d'accord est connu à environ ± 6 points.
- **Pour chaque ligne**, l'annotateur indique :
  - son propre verdict ;
  - pour le feedback :
    - donne-t-il la réponse ? (oui / non)
    - est-il cohérent avec le verdict ? (oui / non)
    - qualité pédagogique : 0, 1 ou 2.
- **Deux annotateurs** relisent les mêmes 30 à 40 lignes, pour mesurer l'accord entre humains.
- **Deux passages** : un sur le pilote, pour corriger le prompt, puis un sur le dataset complet, pour obtenir le chiffre final du rapport.

### Métriques

- **Taux d'accord** : pourcentage de lignes où l'annotateur et le label imposé coïncident.
- **Matrice de confusion** : montre où se situent les désaccords. La frontière `PARTIEL` / `INCORRECT` est la plus délicate.
- **Kappa de Cohen** : l'accord corrigé du hasard.

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

$p_o$ est l'accord observé et $p_e$ l'accord attendu par hasard, calculé à partir de la fréquence à laquelle chacun utilise chaque verdict.

*Exemple :* 85 accords sur 100 avec 4 verdicts équilibrés donnent $p_o = 0{,}85$ et $p_e \approx 0{,}25$, donc $\kappa \approx 0{,}80$.

| Kappa | Lecture (Landis & Koch) |
|---|---|
| < 0,20 | Très faible |
| 0,21 – 0,40 | Faible |
| 0,41 – 0,60 | Modéré |
| 0,61 – 0,80 | Bon |
| > 0,80 | Excellent |

On calcule deux kappas :
1. **Humain contre labels imposés** : est-ce que le dataset est fiable ?
2. **Humain contre humain** : est-ce que les fiches sont assez claires ? Si deux humains ne sont pas d'accord entre eux, c'est la fiche qu'il faut préciser. Ce kappa sert aussi de plafond réaliste pour le premier.

### Seuils d'acceptation (fixés avant l'audit)

- Taux d'accord **≥ 90 %** et kappa **≥ 0,70** entre l'humain et les labels imposés.
- **Aucun feedback** de l'échantillon ne donne la réponse.
- Si un seuil n'est pas atteint, on corrige la fiche ou le prompt pour les questions concernées, puis on regénère et on refait l'audit.

## 6. Points encore ouverts

Ce dataset de 1 003 lignes est le **dataset de référence** pour l'entraînement QLoRA.

1. **Split train / val / test (≈ 65 / 18 / 18 %)** : il est fait par question entière, pour que val et test mesurent la généralisation à des questions jamais vues à l'entraînement. Un découpage ligne par ligne aurait donné pile 70 / 15 / 15, mais avec les mêmes questions en train et en test, donc un score gonflé.

   | Split | Questions | Lignes |
   |---|---|---|
   | `train` | 11 (6 philosophie, 5 maths) | 649 |
   | `val` | `dilemme-du-tramway_q2`, `monty-hall_q3`, `plier-une-feuille-42-fois_q2` | 177 |
   | `test` | `bateau-de-thesee_q3`, `doute-de-descartes_q1`, `infinis-plus-grands-que-l-infini_q2` | 177 |

   Chaque concept garde au moins une question en `train`. Le test contient une Q1, une Q2 et une Q3 de trois concepts différents, pour mesurer aussi l'usage de l'historique. Les verdicts (16 / 16 / 16 / 11 par question) et les essais ont les mêmes proportions dans les trois splits. Le test reste généré par les mêmes modèles que l'entraînement ; `evaluation/cases_v0.1.jsonl` porte sur `monty-hall_q2`, qui est en `train`, et ne doit pas servir de test final.
