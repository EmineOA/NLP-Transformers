# Brainmaxing — Backlog complet (Epics, User Stories, Tâches techniques, Sprints)

> **Notion est la référence.** L'équipe a reporté ce backlog dans Notion (page « ED - Tech / Brainmaxing | NLP »), puis l'a modifié. Ce fichier est aligné sur Notion au 2026-10-03. En cas d'écart, c'est Notion qui fait foi.
>
> **v4** — alignement sur Notion : répartition de l'équipe par blocs, 4 contrats communs (fiche concept, sortie de l'agent, fonction d'inférence, cas de test), format de sortie simplifié (`[VERDICT: X]` + feedback), hors périmètre = `verdict: null`.
>
> **v3** — intègre les suggestions de l'équipe : Definition of Done (commune + Sprint 1), contrat de sortie figé en S1, tests négatifs et test d'historique en S1, versionnement, reproductibilité, mesure mémoire/temps, matrice de traçabilité, tests rejoués avant/après le mini-LoRA.

> Mini-projet 2 : agent conversationnel. Équipe de 6, **3 sprints d'une semaine**, pas de marge.
> Agent tuteur : l'utilisateur choisit un concept, le lit, répond à l'écrit à 3 questions ; l'agent évalue (PARFAIT / PARTIEL / INCORRECT) et donne des indices.

## Décisions proposées

| Sujet | Décision |
|---|---|
| Domaine | **1 domaine, 2 maximum** (ex. philosophie) — *domaine exact à choisir* |
| Concepts | ≈ 6 concepts × 3 questions (définition, exemple, application) — *nombre à confirmer* |
| Contenu | Fiches concept en JSON injectées dans le prompt (pas de RAG ; RAG = amélioration possible dans le rapport) |
| Sortie du modèle | **Contrat figé en S1** : `[VERDICT: X]` en première ligne + feedback libre (voir [Contrats communs](#contrats-communs)) |
| Évaluation d'une réponse | PARFAIT / PARTIEL / INCORRECT selon les points clés (règles : US2) ; hors périmètre = pas de verdict (`verdict: null`) |
| Essais | 3 essais max par question, **comptés par Gradio** (pas par le modèle), puis points clés affichés et question suivante |
| Refus | 5 cas : « donne la réponse », hors domaine, concept absent des fiches, « fais mon devoir », réponse vide/sans rapport |
| Dataset | ≈ 300 dialogues : 25 % parfait · 30 % partiel · 30 % incorrect · 15 % refus ; split 80 / 10 / 10 |
| Évaluation du modèle | **Deux niveaux** : métrique automatique (accuracy du verdict, format, refus) + grille humaine (critères du sujet) |
| Versionnement | Prompt, fiches, dataset, adaptateur et résultats versionnés (`v0.1` = S1, `v0.2` = S2, `v1.0` = S3) |
| Modèle | `Qwen2.5-3B-Instruct` en QLoRA 4-bit (repli : `Qwen2.5-1.5B-Instruct`, décidé avec la mesure de T5.1) |
| Matériel | Google Colab (T4, fp16) |
| Interface | Gradio lancée depuis le notebook |
| Organisation | Un responsable par bloc (voir [Équipe](#équipe)). Un élément est validé par quelqu'un d'autre que son auteur |
| Méthode | MVP de bout en bout en Sprint 1, puis passage à l'échelle (S2), puis finition (S3) |

## User Stories vs tâches techniques

- **User Story (US)** : un besoin de l'**utilisateur final** (l'apprenant). Forme « En tant qu'utilisateur, je veux… pour… », critères Given / When / Then.
  Une US décrit **l'expérience de l'utilisateur** : ce qu'il fait, ce qu'il voit, ce qu'il obtient. Aucun détail technique (tags, parseur, compteur…) dans ses critères : ils vont dans les tâches, ou dans une ligne *Implémentation* sous la US.
- **Tâche technique (T)** : travail nécessaire au projet mais sans valeur directe pour l'utilisateur (observer les logits, entraîner le LoRA, écrire le rapport…). Forme : objectif + critères de fin.
- Les deux sont rangés dans les mêmes epics et les mêmes sprints.

## Équipe

Répartition décidée sur Notion. Elle remplace les binômes A / B / C de la v3.

| Bloc Notion | Epics | Responsable(s) |
|---|---|---|
| A — Pédagogie / comportement | E1 | Jalis + Nikko |
| B — Moteur LLM / baseline | E2 · E3 | Emine |
| C — Dataset / QLoRA | E4 · E5 | Simon |
| D — Évaluation | E6 · matrice de traçabilité | Massi |
| E — Application Gradio | E7 | **à attribuer** (Nikko ou Anthusan) |
| F — Rendu / reproductibilité | E8 | **à attribuer** (Nikko ou Anthusan) · tous pour la relecture du dataset (T4.2) |

Dans les tableaux de sprint plus bas, la colonne « Binôme » utilise encore les lettres de la v3 : A = blocs B + E, B = bloc C, C = blocs A + D.

## Vue d'ensemble

| Epic | Étape du sujet | Pts | Sprint 1 (MVP v0.1) | Sprint 2 (v0.2) | Sprint 3 (v1.0) |
|---|---|---|---|---|---|
| E1 Comportement de l'agent | 1 | 2 | US2 ★ · US3 ★ · US4 ★ · US5 ★ · US7 ★ · T1.1 ★ · T1.2 ★ · T1.3 · T1.4 ★ | — | — |
| E2 Observation du modèle | 2 | 3 | T2.1 ★ · T2.2 | — | — |
| E3 Baseline conversationnelle | 3 | 3 | T3.1 ★ | T3.2 | — |
| E4 Dataset de dialogues | 4 | 2 | T4.1 ★ | T4.2 | — |
| E5 Adaptation LoRA / QLoRA | 5 | 4 | T5.1 ★ | T5.2 | T5.3 |
| E6 Évaluation avant / après | 6 | 3 | T6.1 ★ | T6.2 | T6.3 |
| E7 Interface Gradio | 7 | 3* | US8 ★ | US1 · US6 · US9 | T7.1 |
| E8 Livrables et rapport | — | 3* | T8.1 ★ | T8.2 | T8.3 |

★ = chemin critique du MVP. \* Les 3 pts « interface, reproductibilité, rendu » sont partagés entre E7 et E8.

**Total : 8 epics · 9 User Stories · 20 tâches techniques = 29 éléments** — S1 : 17 · S2 : 8 · S3 : 4.

---

# Règles communes

## Definition of Done commune

Valable pour **chaque** US et **chaque** tâche, dans tous les sprints. Un élément passe en « Done » seulement si :

1. Ses critères d'acceptation (US) ou de fin (tâche) sont remplis.
2. Il est **validé par une autre personne que son auteur**.
3. **Reproductibilité** : cette autre personne a relancé la section concernée du notebook sur Colab et obtenu le même résultat (ou un résultat équivalent pour la génération).
4. Les artefacts produits sont **versionnés** (voir ci-dessous) et rangés dans le Drive partagé.
5. Si l'élément change un comportement de l'agent, la **matrice de traçabilité** est mise à jour.

## Versionnement

| Artefact | Nom | Exemple |
|---|---|---|
| Fiches concept | `fiches_vX.Y.json` | `fiches_v0.1.json` |
| Prompt système | `prompt_vX.Y.txt` | `prompt_v0.1.txt` |
| Dataset | `dataset_vX.Y.jsonl` (+ `_train`, `_val`, `_test`) | `dataset_v0.2_train.jsonl` |
| Adaptateur LoRA | `adapter_vX.Y/` | `adapter_v0.1/` |
| Résultats d'évaluation | `eval_vX.Y_<modèle>.csv` | `eval_v0.1_baseline.csv` |

- `v0.1` = Sprint 1, `v0.2` = Sprint 2, `v1.0` = Sprint 3. Correction dans un sprint : `v0.2.1`, etc.
- Chaque fichier de résultats indique les versions utilisées : prompt, fiches, dataset, adaptateur.
- On ne modifie jamais un fichier versionné : on crée la version suivante.

## Contrats communs

Ils sont définis sur Notion et figés en Sprint 1 (T1.4). Un contrat fixe seulement le format des données échangées entre les blocs. Chaque bloc code comme il veut, tant qu'il respecte ces formats.

### 1. Fiche concept

```json
{
  "id": "bio_photosynthese",
  "matiere": "Biologie",
  "concept": "Photosynthèse",
  "explication": "Explication courte du concept.",
  "questions": [
    {
      "id": "bio_photosynthese_q1",
      "type": "definition",
      "enonce": "Qu'est-ce que la photosynthèse ?",
      "points_cles": ["utilisation de l'énergie lumineuse", "utilisation du dioxyde de carbone", "production de matière organique"]
    }
  ]
}
```

- 1 fiche = 1 concept, 3 questions dans l'ordre définition → exemple → application.
- 2 à 4 points clés par question.
- Chaque fiche et chaque question a un identifiant unique.

### 2. Sortie de l'agent

La première ligne est toujours le verdict, suivie d'un feedback libre :

```
[VERDICT: PARTIEL]
Ton explication contient un élément correct. Pense aussi au rôle de la lumière.
```

Verdicts autorisés : `PARFAIT`, `PARTIEL`, `INCORRECT`. Après parsing :

```json
{"verdict": "PARTIEL", "feedback": "...", "raw_response": "[VERDICT: PARTIEL]\n...", "valid_format": true}
```

- **Hors périmètre** : `{"verdict": null, "feedback": "Cette demande sort du périmètre de l'agent.", "valid_format": true}`.
- **Sortie mal formée** : `valid_format: false`. Gradio affiche la sortie telle quelle et ne fait pas avancer le parcours.
- **Parcours géré par Gradio, pas par le modèle** :
  - PARFAIT → question suivante.
  - PARTIEL ou INCORRECT, moins de 3 essais → nouvel essai.
  - PARTIEL ou INCORRECT, 3 essais → points clés affichés, puis question suivante.

### 3. Fonction d'inférence

```python
generate_response(messages, concept, question, model_version="baseline")  # ou "lora"
# -> {"verdict": ..., "feedback": ..., "raw_response": ..., "valid_format": ...}
```

Gradio et les tests passent uniquement par cette fonction. Ils ne dépendent pas de l'implémentation du modèle.

### 4. Cas de test

```json
{
  "id": "test_001",
  "concept_id": "bio_photosynthese",
  "question_id": "bio_photosynthese_q1",
  "history": [],
  "user_input": "La photosynthèse permet à la plante d'utiliser la lumière.",
  "expected": {"verdict": "PARTIEL", "hors_perimetre": false},
  "tags": ["biologie", "partiel", "single_turn"]
}
```

Les tests couvrent les 3 verdicts, le multi-tours, l'historique, le hors périmètre, les refus et les erreurs de format. Ils restent indépendants du dataset d'entraînement.

## Matrice de traçabilité

Relie chaque comportement attendu à sa règle, ses exemples d'entraînement, ses tests et son effet dans l'interface. Tenue par le bloc D (Massi), mise à jour à chaque changement de comportement (DoD commune, point 5).

| Comportement | US | Règle (où) | Exemples dataset | Tests E6 | Effet dans Gradio |
|---|---|---|---|---|---|
| Réponse parfaite | US3 | Contrat 2 + prompt | catégorie PARFAIT (25 %) | N1 | Question suivante |
| Réponse incomplète | US4 | Contrat 2 + prompt | PARTIEL « incomplet » | N2 | Nouvel essai |
| Imprécision mineure | US4 | Définition US2 | PARTIEL « imprécision » | (S2) | Nouvel essai |
| Erreur majeure | US5 | Définition US2 | catégorie INCORRECT (30 %) | N3 | Nouvel essai |
| Prise en compte de l'historique | US2 | T3.1 (historique N tours) | dialogues multi-tours | N4 | 2e essai évalué avec le 1er |
| « Donne-moi la réponse » | US7 | Contrat 2 (`verdict: null`) | refus (15 %) | N5 | Pas d'avance |
| Concept absent des fiches | US7 | Contrat 2 (`verdict: null`) | refus | X2 | Pas d'avance, pas d'invention |
| Hors périmètre déguisé | US7 | Contrat 2 (`verdict: null`) | refus | X3 | Pas d'avance |
| Sortie mal formée | T1.4 | Parseur → `valid_format: false` | — | X1 | Affichage brut, pas d'avance |
| 3 essais ratés | US6 | Code Gradio + fiche | — (non appris) | (S2) | Points clés affichés, question suivante |

Les tests N1–N5 et X1–X3 sont définis dans T6.1. La colonne « Tests E6 » est complétée en S2 avec les 20 tests de T6.2.

---

# Sprints

## Sprint 1 — MVP v0.1

**Objectif :** `agent.ipynb` tourne de bout en bout sur Colab avec 1 concept : fiche → baseline → 30 dialogues → LoRA 5 min → tests → Gradio.

**Ordre imposé :**
1. **Jour 2** : format des fiches (T1.2), contrat de sortie (T1.4) et format des dialogues (T4.1) validés par tous.
2. Les tests de T6.1 sont passés sur la baseline **avant** le mini-LoRA (T5.1), puis **rejoués à l'identique** sur `adapter_v0.1`.

### Definition of Done du Sprint 1

Le Sprint 1 est terminé seulement si **tout** ceci est vrai :

- [ ] Toutes les tâches et US ★ sont « Done » (DoD commune respectée).
- [ ] `agent.ipynb` s'exécute de bout en bout (« Exécuter tout ») sur un Colab vierge, **lancé par une personne qui n'a pas écrit le code**.
- [ ] Le contrat de sortie `v0.1` est figé et documenté.
- [ ] Les 8 tests de S1 (5 nominaux + 3 négatifs, dont 1 test d'historique) sont passés sur la baseline, puis rejoués sur `adapter_v0.1` : résultats dans `eval_v0.1_baseline.csv` et `eval_v0.1_lora.csv`.
- [ ] Le coût mémoire et temps du mini-LoRA est mesuré, et la décision 3B / 1.5B est prise.
- [ ] Les artefacts `fiches_v0.1`, `prompt_v0.1`, `dataset_v0.1`, `adapter_v0.1` existent dans le Drive.
- [ ] La matrice de traçabilité `v0.1` est remplie pour les lignes N1–N5 et X1–X3.
- [ ] Gradio répond sur le concept pilote en respectant le contrat de sortie.

| # | Titre | Type | Epic | Binôme | MVP |
|---|---|---|---|---|---|
| US2 | Recevoir un verdict sur ma réponse | US | E1 | C | ★ |
| US3 | Réponse parfaite : confirmation et suite | US | E1 | C | ★ |
| US4 | Réponse partielle : petits indices | US | E1 | C | ★ |
| US5 | Réponse incorrecte : correction et gros indices | US | E1 | C | ★ |
| US7 | Demande hors périmètre : refus ou redirection | US | E1 | C | ★ |
| US8 | Discuter avec l'agent dans une interface | US | E7 | A | ★ |
| T1.1 | Identité de l'agent : cible, rôle, périmètre, 5 conversations | Tâche | E1 | C | ★ |
| T1.2 | Format des fiches + fiche pilote | Tâche | E1 | C | ★ |
| T1.3 | Toutes les fiches du domaine | Tâche | E1 | C | |
| T1.4 | Contrat de sortie + parseur | Tâche | E1 | C | ★ |
| T2.1 | Modèle 4-bit + tokens, logits, softmax | Tâche | E2 | A | ★ |
| T2.2 | Argmax, température, top-k, top-p + lien avec les TP | Tâche | E2 | A | |
| T3.1 | Baseline : prompt système + fiche + chat template + historique limité | Tâche | E3 | A | ★ |
| T4.1 | Format JSONL + script de génération + 30 dialogues pilotes | Tâche | E4 | B | ★ |
| T5.1 | Pipeline QLoRA minimal + mesure mémoire/temps | Tâche | E5 | B | ★ |
| T6.1 | Grille + métrique + 8 tests (5 nominaux, 3 négatifs) avant/après mini-LoRA | Tâche | E6 | C | ★ |
| T8.1 | Squelette du notebook + règles de travail + versionnement | Tâche | E8 | Tous | ★ |

## Sprint 2 — v0.2 (passage à l'échelle)

**Objectif :** vrai dataset, vrai entraînement, parcours complet dans Gradio, début du rapport.

**Ordre imposé :** T6.2 (tests) et T4.2 (dataset) **avant** T5.2 (entraînement).

**Fini quand :** DoD commune respectée pour chaque élément ; `adapter_v0.2` entraîné sur `dataset_v0.2` ; les 20 tests passés sur la baseline et sur `adapter_v0.2` ; matrice de traçabilité complétée avec les tests de T6.2.

| # | Titre | Type | Epic | Binôme |
|---|---|---|---|---|
| US1 | Choisir un concept et le découvrir | US | E7 | A |
| US6 | Bloqué après 3 essais : voir les points clés et continuer | US | E7 | A |
| US9 | Enchaîner les questions d'un concept, puis passer au suivant | US | E7 | A |
| T3.2 | Comparaison de 2 prompts + paramètres justifiés + réponses de référence | Tâche | E3 | A |
| T4.2 | ≈ 300 dialogues équilibrés, relus, split train/val/test | Tâche | E4 | B + tous |
| T5.2 | Entraînement complet + courbes de loss + meilleur adaptateur | Tâche | E5 | B |
| T6.2 | 20 tests (dont historique et refus) + métrique automatique | Tâche | E6 | C |
| T8.2 | Rapport partie 1 : cas d'usage + données | Tâche | E8 | Tous |

## Sprint 3 — v1.0 (rendu)

**Objectif :** évaluation finale, rapport, notebook propre. **Gel des fonctionnalités en milieu de semaine.**

**Fini quand :** tous les livrables du sujet sont prêts ; `agent.ipynb` ré-exécuté sur un Colab vierge par une personne qui n'a pas écrit le code.

| # | Titre | Type | Epic | Binôme |
|---|---|---|---|---|
| T5.3 | Hyperparamètres documentés + adaptateur publié | Tâche | E5 | B |
| T6.3 | Évaluation finale à deux niveaux + analyse de 5 cas | Tâche | E6 | C |
| T7.1 | Gradio branché sur le LoRA | Tâche | E7 | A |
| T8.3 | Rapport final + README + ré-exécution sur Colab vierge | Tâche | E8 | Tous |

---

# Epics détaillées

## E1 · Comportement de l'agent

**Étape 1 du sujet — 2 pts · Bloc A (Jalis + Nikko)**
Définir le rôle de l'agent, comment il évalue, ce qu'il refuse, le contenu des concepts et le contrat de sortie.
Les US d'E1 sont implémentées par le prompt (T3.1) et apprises par le dataset (T4.1, T4.2) ; le bloc A écrit les règles et les exemples, et vérifie le résultat.

### User Stories

#### US2 — Recevoir un verdict sur ma réponse ★ S1
*En tant qu'utilisateur, je veux une évaluation de chacune de mes réponses, pour savoir ce que j'ai compris et ce qui me manque.*

Regroupe la US Notion existante « Mécanisme de l'évaluation ».

| Verdict | Règle |
|---|---|
| PARFAIT | Tous les points clés présents, aucune erreur |
| PARTIEL | Compréhension globalement correcte, mais **incomplète** (points clés manquants) **ou avec une imprécision mineure** |
| INCORRECT | Aucun point clé correct, **ou** une erreur majeure |

- **Imprécision mineure** : vocabulaire approximatif ou détail inexact qui ne contredit aucun point clé.
- **Erreur majeure** : affirmation qui contredit un point clé ou montre un contresens sur le concept.
- Une réponse avec plusieurs bons éléments et une seule imprécision mineure est donc **PARTIEL**, pas INCORRECT.

Critères d'acceptation :
- **Given** je viens de lire une question
- **When** j'envoie ma réponse
- **Then** je vois clairement si ma réponse est parfaite, partielle ou incorrecte
- **And** je vois ce que j'ai bien compris, ce qui me manque, et ce que je dois faire ensuite
- **And** si je réessaie, l'agent se souvient de ma réponse précédente et ne me redemande pas ce que j'ai déjà dit
- **And** l'agent ne me donne jamais la réponse complète à ma place

*Implémentation : [contrat 2](#2-sortie-de-lagent) (T1.4), prompt et historique (T3.1).*

Priorité 10 · Difficulté 5

#### US3 — Réponse parfaite ★ S1
*En tant qu'utilisateur, quand ma compréhension est parfaite, je veux que l'agent me le confirme et me propose la suite.*

US Notion existante « cas parfait ».
- **Given** ma réponse contient tous les points attendus, sans erreur
- **When** je l'envoie
- **Then** je vois que ma réponse est parfaite
- **And** l'agent me rappelle en une phrase ce que j'ai bien compris
- **And** je passe à la question suivante

*Implémentation : `[VERDICT: PARFAIT]` (contrat 2) ; Gradio passe à la question suivante.*

Priorité 8 · Difficulté 2

#### US4 — Réponse partielle ★ S1
*En tant qu'utilisateur, quand ma réponse est partielle, je veux que l'agent m'aide à la compléter sans me donner la réponse.*

US Notion existante « cas partielle ».
- **Given** ma réponse est globalement juste, mais incomplète ou un peu imprécise
- **When** je l'envoie
- **Then** je vois que ma réponse est partielle
- **And** l'agent me dit ce qui est déjà juste, et corrige mon imprécision s'il y en a une
- **And** pour chaque point manquant, il me pose **une petite question-indice**, sans me donner la réponse
- **And** je suis invité à compléter ma réponse

*Implémentation : `[VERDICT: PARTIEL]` (contrat 2) ; Gradio propose un nouvel essai.*

Priorité 8 · Difficulté 3

#### US5 — Réponse incorrecte ★ S1
*En tant qu'utilisateur, quand ma réponse est fausse, je veux que l'agent corrige mon erreur et me guide plus fortement.*

US Notion existante « cas inexacte ».
- **Given** ma réponse ne contient aucun point juste, ou contient une erreur importante
- **When** je l'envoie
- **Then** je vois que ma réponse est incorrecte
- **And** l'agent m'explique d'abord mon erreur, sans me juger
- **And** il me donne **des indices plus directs** qui me mettent sur la bonne piste, sans rédiger la réponse
- **And** je suis invité à réessayer

*Implémentation : `[VERDICT: INCORRECT]` (contrat 2) ; Gradio propose un nouvel essai.*

Priorité 8 · Difficulté 3

#### US7 — Demande hors périmètre ★ S1
*En tant qu'utilisateur, je veux que l'agent me dise clairement ce qu'il ne fait pas, pour ne pas recevoir de réponse inventée.*

| Cas | Réponse attendue |
|---|---|
| « Donne-moi la réponse » | Refuse, propose un indice à la place |
| Question hors du domaine | Redirige vers les concepts disponibles |
| Concept du domaine absent des fiches | Dit qu'il ne le connaît pas encore, n'improvise pas |
| « Fais mon devoir » | Refuse, propose d'expliquer le concept |
| Réponse vide ou sans rapport | Encourage et reformule la question |

- **Given** je fais une demande de l'un des 5 cas
- **When** je l'envoie
- **Then** l'agent m'explique poliment pourquoi il ne peut pas répondre
- **And** il me propose ce que je peux faire à la place (voir tableau)
- **And** il n'invente aucun contenu

*Implémentation : `verdict: null` (contrat 2) ; aucun essai consommé, pas d'avance.*

Priorité 9 · Difficulté 3

### Tâches techniques

#### T1.1 — Identité de l'agent ★ S1
Rédiger la présentation de l'agent demandée par le sujet (étape 1).
- Contient : utilisateur cible, rôle, demandes traitées, demandes refusées, 5 conversations types
- Réutilisée dans le notebook, dans Gradio (présentation) et dans le rapport

Priorité 10 · Difficulté 3

#### T1.2 — Format des fiches + fiche pilote ★ S1
Définir le format commun utilisé par le dataset, les tests et Gradio.
- Schéma : `domaine`, `concept`, `explication`, `questions[]` avec pour chaque question `type` (définition / exemple / application), `énoncé`, `points_clés[]` (2 à 4)
- Une fiche pilote complète (1 concept, 3 questions) dans `fiches_v0.1.json`
- Schéma validé par tous au jour 2

Priorité 10 · Difficulté 3

#### T1.3 — Toutes les fiches du domaine · S1
Écrire toutes les fiches avant la fin du Sprint 1, pour lancer le dataset complet dès le début du Sprint 2.
- ≈ 6 concepts × 3 questions dans le domaine choisi (ou 3 + 3 si 2 domaines)
- Chaque point clé vérifié par une autre personne que l'auteur
- Livré dans `fiches_v0.2.json`

Priorité 8 · Difficulté 4

#### T1.4 — Contrat de sortie + parseur ★ S1
Figer les [contrats communs](#contrats-communs), surtout la sortie de l'agent (contrat 2).
- Format documenté avec un exemple par cas : PARFAIT, PARTIEL, INCORRECT, hors périmètre
- Fonction de parsing qui renvoie `{verdict, feedback, raw_response, valid_format}`
- Le parseur ne plante jamais, même sur une sortie vide ou mal formée
- Validé par tous au jour 2 ; toute modification après S1 = nouvelle version du prompt et du dataset

Priorité 10 · Difficulté 3

---

## E2 · Observation du modèle

**Étape 2 du sujet — 3 pts · Binôme A**

#### T2.1 — Modèle, tokens, logits, softmax ★ S1
- `Qwen2.5-3B-Instruct` chargé en 4-bit avec `transformers` sur Colab T4
- Sur un exemple court, le notebook affiche et commente : identifiants de tokens, forme des logits, distribution du prochain token après softmax (top 10)

Priorité 9 · Difficulté 4

#### T2.2 — Stratégies de décodage + lien avec les TP · S1
- Même prompt généré avec argmax, plusieurs températures, top-k et top-p ; sorties comparées et commentées
- Paragraphe reliant les observations aux embeddings, à l'attention causale et à l'unembedding

Priorité 7 · Difficulté 4

---

## E3 · Baseline conversationnelle

**Étape 3 du sujet — 3 pts · Binôme A**

#### T3.1 — Baseline ★ S1
Agent sans entraînement, point de comparaison pour le LoRA.
- Message système (rôle + règles de verdict + contrat de sortie + fiche injectée) dans `prompt_v0.1.txt`
- `apply_chat_template`, historique limité aux N derniers tours (N documenté)
- Sur la fiche pilote, la sortie respecte le contrat (vérifié avec le parseur de T1.4)
- **Test d'historique** : sur un 2e essai qui complète le 1er, l'agent tient compte de la réponse précédente (test N4)

Priorité 10 · Difficulté 5

#### T3.2 — Comparaison de prompts · S2
- Au moins 2 prompts comparés (ou zero-shot contre few-shot), notés avec les deux niveaux d'évaluation
- Paramètres de génération justifiés
- Réponses de la meilleure configuration sauvegardées comme référence ; prompt retenu = `prompt_v0.2.txt`

Priorité 8 · Difficulté 4

---

## E4 · Dataset de dialogues

**Étape 4 du sujet — 2 pts · Binôme B (+ relecture par tous)**

#### T4.1 — Format, script, 30 dialogues pilotes ★ S1
- Format JSONL `{"messages": [system, user, assistant, ...]}`, validé par tous au jour 2
- Réponses de l'assistant **au format du contrat de sortie** (vérifiées avec le parseur de T1.4)
- Script de génération reproductible
- 30 dialogues pilotes relus à la main dans `dataset_v0.1.jsonl` : les 3 verdicts, au moins 3 refus, au moins 2 dialogues multi-tours

Priorité 10 · Difficulté 5

#### T4.2 — Dataset complet · S2
- ≈ 300 dialogues construits à partir de `fiches_v0.2.json`
- Répartition : **25 % PARFAIT · 30 % PARTIEL · 30 % INCORRECT · 15 % refus**
- Les PARTIEL incluent des cas « incomplet » **et** des cas « imprécision mineure » ; une part des dialogues est multi-tours
- 100 % des réponses passent le parseur (`valid_format: true`)
- Chaque dialogue relu par un humain, sans donnée personnelle
- Split train / validation / test 80 / 10 / 10 dans `dataset_v0.2_*.jsonl` ; aucune requête des 20 tests (T6.2) dans le train

Priorité 10 · Difficulté 7

---

## E5 · Adaptation LoRA / QLoRA

**Étape 5 du sujet — 4 pts · Binôme B**

#### T5.1 — Pipeline QLoRA minimal + mesure mémoire/temps ★ S1
Découvrir dès la semaine 1 les problèmes de Colab (mémoire, format) et chiffrer le vrai entraînement.
- Entraînement court (≈ 5 min) sur `dataset_v0.1.jsonl`, modèle de base gelé
- Loss affichée ; adaptateur sauvegardé dans `adapter_v0.1/` et rechargeable
- **Mesures notées dans le notebook** : pic de mémoire GPU (`torch.cuda.max_memory_allocated`), temps par step, temps total
- **Extrapolation** à ≈ 300 dialogues et plusieurs époques → décision : on garde le 3B, ou on passe au 1.5B (si dépassement mémoire ou durée estimée > 1 h)
- Démarré **après** le passage des tests T6.1 sur la baseline

Priorité 10 · Difficulté 7

#### T5.2 — Entraînement complet · S2
- Entraînement sur `dataset_v0.2_train.jsonl`, validation sur `dataset_v0.2_val.jsonl`
- Courbes de loss train et validation affichées
- Meilleur adaptateur selon la validation sauvegardé dans `adapter_v0.2/` ; arrêt justifié (ex. la loss de validation remonte)

Priorité 10 · Difficulté 8

#### T5.3 — Documentation + publication · S3
- Documentés : modèle de base, quantification, rang LoRA, taux d'apprentissage, taille des lots, nombre d'époques, longueur maximale, mémoire et durée réelles
- Adaptateur final `adapter_v1.0/` livré ou accessible par un lien stable

Priorité 9 · Difficulté 3

---

## E6 · Évaluation avant / après

**Étape 6 du sujet — 3 pts · Binôme C**

L'évaluation a **deux niveaux**, utilisés pour la baseline comme pour le modèle adapté :

| Niveau | Quoi | Comment |
|---|---|---|
| 1 · Automatique | Accuracy du verdict (prédit vs attendu) · taux de sorties conformes au contrat (100 % `valid_format: true` visé) · taux de refus corrects | Script + parseur de T1.4 |
| 2 · Humain | Grille des 5 critères du sujet : pertinence et correction · respect du rôle et du format · prise en compte de l'historique · reconnaissance du hors périmètre · informations inventées | Échelle 0 / 1 / 2 par critère, notée par une personne et vérifiée par une autre |

Le niveau 1 mesure si l'agent classe bien ; le niveau 2 mesure si ses retours sont bons (indices utiles, pas d'invention, ton adapté). **Les deux sont présentés dans le rapport.**

#### T6.1 — Grille, métrique et 8 tests avant/après mini-LoRA ★ S1
- Grille humaine écrite **avant tout entraînement**, avec une échelle explicite
- Script de la métrique automatique (utilise le parseur de T1.4)
- **5 tests nominaux** :

| ID | Test | Attendu |
|---|---|---|
| N1 | Réponse complète et exacte | PARFAIT |
| N2 | Réponse incomplète, sans erreur | PARTIEL |
| N3 | Réponse avec une erreur majeure | INCORRECT |
| N4 | **Historique** : 1er essai partiel, 2e essai qui complète le 1er | PARTIEL puis PARFAIT |
| N5 | « Donne-moi la réponse » | Hors périmètre (`verdict: null`) |

- **3 tests négatifs** (robustesse de l'architecture) :

| ID | Test | Attendu |
|---|---|---|
| X1 | Sortie de modèle sans tag (texte fabriqué, passé directement au parseur et à Gradio) | Parseur → `valid_format: false`, Gradio ne plante pas et n'avance pas |
| X2 | Question sur un concept du domaine absent des fiches | Hors périmètre (`verdict: null`), aucune invention de contenu |
| X3 | Demande hors périmètre déguisée (ex. « rédige-moi une dissertation sur ce concept ») | Hors périmètre (`verdict: null`) |

- **Ordre** : les 8 tests sont passés sur la baseline (`eval_v0.1_baseline.csv`) **avant** T5.1, puis **rejoués à l'identique** sur `adapter_v0.1` (`eval_v0.1_lora.csv`)

Priorité 10 · Difficulté 4

#### T6.2 — 20 tests · S2
- Écrits **avant** l'entraînement complet (T5.2), jamais utilisés à l'entraînement ; reprennent N1–N5 et X1–X3
- Au moins 20 requêtes, dont au moins 3 conversations sur plusieurs tours (historique), 5 refus, et des PARTIEL « imprécision mineure »
- Chaque test a son verdict attendu ; matrice de traçabilité complétée

Priorité 10 · Difficulté 4

#### T6.3 — Évaluation finale · S3
- Baseline et modèle adapté passés sur les 20 tests, aux deux niveaux
- Tableau synthétique : métriques automatiques + scores de la grille par critère, avec les versions utilisées
- Au moins 5 réussites ou erreurs commentées

Priorité 10 · Difficulté 5

---

## E7 · Interface Gradio

**Étape 7 du sujet — partie des 3 pts « interface, reproductibilité, rendu » · Binôme A**

### User Stories

#### US8 — Discuter avec l'agent ★ S1
*En tant qu'utilisateur, je veux une interface de chat simple, pour parler à l'agent.*
- **Given** le notebook lancé
- **When** j'ouvre l'interface
- **Then** je vois une zone de conversation, un champ de saisie, un bouton pour recommencer à zéro, et une courte présentation de ce que fait l'agent et de ses limites
- **And** le verdict de chaque réponse est mis en évidence (ex. ✅ Parfait · 🟡 Partiel · ❌ Incorrect · ⛔ Hors sujet)
- **And** si l'agent répond de façon inattendue, l'interface continue de fonctionner

*Implémentation : texte de T1.1 ; sorties lues par le parseur de T1.4 (une sortie `valid_format: false` est affichée telle quelle).*

Priorité 9 · Difficulté 3

#### US1 — Choisir un concept et le découvrir · S2
*En tant qu'utilisateur, je veux choisir un concept et lire son explication, pour savoir sur quoi je vais être interrogé.*
- **Given** les fiches du domaine
- **When** je choisis un concept dans une liste
- **Then** l'interface affiche son explication puis la première question

Priorité 8 · Difficulté 3

#### US6 — Bloqué après 3 essais · S2
*En tant qu'utilisateur, si je n'y arrive pas après 3 essais, je veux voir la réponse attendue et continuer, pour ne pas rester bloqué.*
- **Given** j'ai répondu 3 fois à la même question sans réussir
- **When** je reçois le 3e retour
- **Then** je vois la réponse attendue, avec une courte explication
- **And** je passe à la question suivante sans rester bloqué

*Implémentation : compteur d'essais tenu par Gradio (pas par le modèle) ; points clés lus dans la fiche.*

Priorité 7 · Difficulté 3

#### US9 — Enchaîner les questions · S2
*En tant qu'utilisateur, je veux enchaîner les questions d'un concept puis passer au suivant, pour apprendre le concept du début à la fin.*
- **Given** je travaille sur un concept
- **When** ma réponse est parfaite
- **Then** je passe à la question suivante (définition → exemple → application)
- **When** ma réponse n'est pas parfaite
- **Then** je peux réessayer, jusqu'à 3 fois (puis US6)
- **And** après la dernière question du concept, on me propose un autre concept

*Implémentation : Gradio décide la suite à partir du verdict (contrat 2) et du compteur d'essais.*

Priorité 8 · Difficulté 5

### Tâches techniques

#### T7.1 — Branchement sur le LoRA · S3
- L'interface utilise l'adaptateur final (`adapter_v1.0/`)
- (Optionnel) sélecteur baseline / LoRA pour la démo

Priorité 8 · Difficulté 2

---

## E8 · Livrables et rapport

**Livrables du sujet — partie des 3 pts « interface, reproductibilité, rendu » · Tous**

#### T8.1 — Squelette du notebook + règles + versionnement ★ S1
- `agent.ipynb` avec les sections 1 à 7 du sujet
- Dossier Drive partagé avec l'arborescence : `fiches/`, `prompts/`, `data/`, `adapters/`, `eval/`
- Convention de [versionnement](#versionnement) appliquée dès le premier fichier
- Chaque bloc travaille dans son propre sous-notebook ; assemblage dans `agent.ipynb` chaque vendredi

Priorité 10 · Difficulté 2

#### T8.2 — Rapport partie 1 · S2
- Sections « cas d'usage » et « données » écrites dès qu'E1 et E4 sont terminées

Priorité 7 · Difficulté 3

#### T8.3 — Rendu final · S3
- `rapport.pdf`, 4 pages maximum hors annexes : cas d'usage, données, choix techniques, comparaison des deux versions (deux niveaux d'évaluation), limites, améliorations (dont le RAG)
- `README.txt` avec les instructions d'exécution
- `agent.ipynb` ré-exécuté de bout en bout sur un Colab vierge par une personne qui n'a pas écrit le code, sorties conservées
- Données (ou script) et adaptateur (ou lien) livrés
- Matrice de traçabilité finale en annexe

Priorité 10 · Difficulté 4

---

# À valider par l'équipe

Chacun coche ou commente. Si une ligne pose problème, on en parle avant de toucher à Notion.

## Retours intégrés

v2 :
- [x] 1 domaine, 2 maximum
- [x] Répartition du dataset 25 / 30 / 30 / 15 (parfait / partiel / incorrect / refus)
- [x] Évaluation à deux niveaux : métrique automatique + grille humaine
- [x] PARTIEL pour une compréhension globalement correcte mais incomplète ou avec imprécision mineure ; INCORRECT réservé aux erreurs majeures
- [x] Séparation User Stories (utilisateur final) / tâches techniques

v3 :
- [x] Definition of Done commune + Definition of Done du Sprint 1
- [x] Contrat de sortie figé en S1 : verdict / compris / manquant / feedback / suite (+ format `[REFUS]`) → nouvelle tâche T1.4
- [x] 3 tests négatifs en S1 (X1 sortie sans tag, X2 concept absent, X3 hors périmètre déguisé)
- [x] Test d'historique en S1 (N4)
- [x] Versionnement de prompt, fiches, dataset, adaptateur et résultats
- [x] Reproductibilité : une autre personne relance la section (DoD commune)
- [x] Mesure mémoire/temps dans le mini-LoRA (T5.1)
- [x] Matrice de traçabilité
- [x] Tests S1 passés avant le mini-LoRA puis rejoués à l'identique après

v3.1 :
- [x] Les critères des US décrivent l'expérience de l'utilisateur (ce qu'il fait, voit, obtient) ; les détails techniques passent dans une ligne *Implémentation*

v4 (décidé sur Notion) :
- [x] Répartition de l'équipe par blocs A à F (voir [Équipe](#équipe))
- [x] 4 contrats communs : fiche concept, sortie de l'agent, fonction d'inférence, cas de test
- [x] Sortie simplifiée : `[VERDICT: X]` + feedback libre (abandon de `Compris` / `Manquant` / `Suite`)
- [x] Hors périmètre = `verdict: null` (abandon du tag `[REFUS]`)
- [x] Parcours (question suivante, nouvel essai, 3 essais) géré par Gradio à partir du verdict

## Points ouverts (v4)

- [ ] **Format brut du hors périmètre** : le contrat 2 donne le JSON parsé (`verdict: null`), mais pas ce que le modèle écrit. Sans marqueur, le parseur ne distingue pas un refus d'une sortie mal formée → proposition : première ligne `[HORS_PERIMETRE]`.
- [ ] **Blocs E (Gradio) et F (rendu)** : à attribuer entre Nikko et Anthusan.
- [ ] **US « Limiter une question à 3 essais »** : assignée à Emine sur Notion, mais c'est de la logique Gradio (bloc E).
- [ ] **US « Gérer le verdict PARFAIT »** : le critère « l'application passe à la question suivante » relève de Gradio, pas de l'agent.
- [ ] **US Notion « Dataset pilote » et « QLoRA minimal »** : assignées à Emine, alors que le bloc C est à Simon.
- [ ] **Nettoyage Notion** : doublon « Définir le format (d'une) fiche concept » ; « Comparer les stratégies de décodage » sans epic ni responsable ; baseline reliée à E3 alors que E3 est intégrée à E2 ; bases des sprints 2 et 3 nommées « Sprint | 1 (1) » et « Sprint | 1 (2) ».

## Structure

- [ ] Les 8 epics (E1 à E8), calquées sur les étapes du sujet
- [ ] Les 9 US et 20 tâches, et leur répartition S1 / S2 / S3
- [ ] Sprint 1 = MVP de bout en bout (1 concept, tout le pipeline), puis passage à l'échelle
- [x] Répartition de l'équipe : voir [Équipe](#équipe) (v4)
- [ ] Les US Notion existantes (« Mécanisme de l'évaluation », « cas parfait », « cas partielle », « cas inexacte ») correspondent à US2, US3, US4, US5

## Contenu

- [ ] Définitions « imprécision mineure » / « erreur majeure » (US2)
- [ ] Les 5 cas de refus (US7)
- [ ] 3 questions par concept : définition → exemple → application
- [ ] Fiches JSON dans le prompt, pas de RAG (RAG cité en amélioration dans le rapport)
- [ ] Modèle `Qwen2.5-3B-Instruct` sur Colab, repli 1.5B selon la mesure de T5.1
- [ ] Un sous-notebook par bloc, assemblage dans `agent.ipynb` chaque vendredi (T8.1)

## Questions sans réponse

1. **Domaine** : lequel ? Un seul, ou deux (ex. philosophie + un autre) ?
2. **Nombre de concepts** : ≈ 6 concepts × 3 questions, ça convient ?
3. **Génération des dialogues avec un gros LLM externe** (ChatGPT, Claude) : autorisée ? → **à demander au prof dès le jour 1.** En attendant, T4.1 (format et gabarits) sert dans les deux cas. Si c'est interdit : dialogues écrits à la main avec gabarits, viser ~200.
4. **Usage de l'IA pour coder** : autorisé. Faut-il le déclarer dans le rapport ? → à demander au prof en même temps que la question 3.
5. **Notion** : garder la base « Sprint | 1 » telle quelle, ou la renommer « User Stories » avec des propriétés « Sprint » (S1 / S2 / S3) et « Type » (US / Tâche) ?
