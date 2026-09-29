# Brainmaxing — Backlog complet (Epics, User Stories, Sprints)

> **PROPOSITION À VALIDER PAR L'ÉQUIPE** — rien n'est encore reporté dans Notion.
> Lisez le document, puis répondez à la section [À valider par l'équipe](#à-valider-par-léquipe) en bas. Une fois tout le monde d'accord, on met Notion à jour.

> Mini-projet 2 : agent conversationnel. Équipe de 6, **3 sprints d'une semaine**, pas de marge.
> Agent tuteur : l'utilisateur choisit une matière, lit un concept, répond à l'écrit à 2–3 questions ; l'agent évalue (PARFAIT / PARTIEL / INCORRECT) et donne des indices.

## Décisions proposées

| Sujet | Décision |
|---|---|
| Matières | Science (maths, physique, bio, chimie), Philosophie, Histoire → 6 domaines |
| Concepts | 1 concept par domaine = 6 concepts × 3 questions (définition, exemple, application) — *à confirmer* |
| Contenu | Fiches concept en JSON injectées dans le prompt (pas de RAG ; RAG = amélioration possible dans le rapport) |
| Évaluation | Points clés par question. Verdict en 1re ligne : `[VERDICT: PARFAIT]`, `[VERDICT: PARTIEL]`, `[VERDICT: INCORRECT]` |
| Essais | 3 essais max par question, puis l'agent montre les points clés et passe à la suite |
| Refus | 5 cas : « donne la réponse », hors matières, concept absent des fiches, « fais mon devoir », réponse vide/sans rapport |
| Modèle | `Qwen2.5-3B-Instruct` en QLoRA 4-bit (repli : `Qwen2.5-1.5B-Instruct`) |
| Matériel | Google Colab (T4, fp16) |
| Interface | Gradio lancée depuis le notebook |
| Organisation | Pas de rôles formels. Une US est validée par quelqu'un d'autre que son auteur |
| Méthode | MVP de bout en bout en Sprint 1, puis passage à l'échelle (S2), puis finition (S3) |

## Binômes

| Binôme | Epics |
|---|---|
| A | E2 Observation · E3 Baseline · E7 Gradio |
| B | E4 Dataset · E5 LoRA |
| C | E1 Comportement · E6 Évaluation |
| Tous | E8 Livrables · relecture du dataset (US 4.2) |

## Vue d'ensemble

| Epic | Étape du sujet | Pts | Sprint 1 (MVP v0.1) | Sprint 2 (v0.2) | Sprint 3 (v1.0) |
|---|---|---|---|---|---|
| E1 Comportement de l'agent | 1 | 2 | 1.1 ★ · 1.2 ★ · 1.3 ★ · 1.4 ★ · 1.5 | — | — |
| E2 Observation du modèle | 2 | 3 | 2.1 ★ · 2.2 | — | — |
| E3 Baseline conversationnelle | 3 | 3 | 3.1 ★ | 3.2 | — |
| E4 Dataset de dialogues | 4 | 2 | 4.1 ★ | 4.2 | — |
| E5 Adaptation LoRA / QLoRA | 5 | 4 | 5.1 ★ | 5.2 | 5.3 |
| E6 Évaluation avant / après | 6 | 3 | 6.1 ★ | 6.2 | 6.3 |
| E7 Interface Gradio | 7 | 3* | 7.1 ★ | 7.2 | 7.3 |
| E8 Livrables et rapport | — | 3* | 8.1 ★ | 8.2 | 8.3 |

★ = chemin critique du MVP. \* Les 3 pts « interface, reproductibilité, rendu » sont partagés entre E7 et E8.

**Total : 8 epics, 23 US** — S1 : 13 · S2 : 6 · S3 : 4.

---

# Sprints

## Sprint 1 — MVP v0.1

**Objectif :** `agent.ipynb` tourne de bout en bout sur Colab avec 1 concept : fiche → baseline → 30 dialogues → LoRA 5 min → 5 tests → Gradio.

**Règle :** le format des fiches (1.2) et le format des dialogues (4.1) sont validés par tous **au jour 2**.

**Fini quand :** « Exécuter tout » marche sur Colab et Gradio répond sur le concept pilote avec un `[VERDICT]`.

| # | User Story | Epic | Binôme | MVP |
|---|---|---|---|---|
| 1.1 | Identité de l'agent : utilisateur cible, rôle, périmètre, 5 conversations types | E1 | C | ★ |
| 1.2 | Format des fiches concept + fiche pilote | E1 | C | ★ |
| 1.3 | Évaluation en 3 verdicts + tag `[VERDICT]` + 3 essais max | E1 | C | ★ |
| 1.4 | Refus des 5 cas hors périmètre | E1 | C | ★ |
| 1.5 | 6 fiches complètes | E1 | C | |
| 2.1 | Modèle 4-bit + tokens, logits, softmax | E2 | A | ★ |
| 2.2 | Argmax, température, top-k, top-p + lien avec les TP | E2 | A | |
| 3.1 | Baseline : prompt système + fiche + chat template + historique limité | E3 | A | ★ |
| 4.1 | Format JSONL + script de génération + 30 dialogues pilotes | E4 | B | ★ |
| 5.1 | Pipeline QLoRA minimal | E5 | B | ★ |
| 6.1 | Grille d'évaluation + 5 premiers tests | E6 | C | ★ |
| 7.1 | Gradio v0.1 | E7 | A | ★ |
| 8.1 | Squelette du notebook + règles de travail | E8 | Tous | ★ |

## Sprint 2 — v0.2 (passage à l'échelle)

**Objectif :** vrai dataset, vrai entraînement, parcours complet dans Gradio, début du rapport.

**Ordre imposé :** 6.2 (tests) et 4.2 (dataset) **avant** 5.2 (entraînement).

| # | User Story | Epic | Binôme |
|---|---|---|---|
| 3.2 | Comparaison de 2 prompts + paramètres justifiés + réponses de référence | E3 | A |
| 4.2 | ~300 dialogues équilibrés, relus, split train/val/test | E4 | B + tous |
| 5.2 | Entraînement complet + courbes de loss + meilleur adaptateur | E5 | B |
| 6.2 | 20 tests (dont historique et refus) + score automatique | E6 | C |
| 7.2 | Parcours complet : matière → concept → questions → 3 essais | E7 | A |
| 8.2 | Rapport partie 1 : cas d'usage + données | E8 | Tous |

## Sprint 3 — v1.0 (rendu)

**Objectif :** évaluation finale, rapport, notebook propre. **Gel des fonctionnalités en milieu de semaine.**

| # | User Story | Epic | Binôme |
|---|---|---|---|
| 5.3 | Hyperparamètres documentés + adaptateur publié | E5 | B |
| 6.3 | Comparaison baseline / LoRA + analyse de 5 cas | E6 | C |
| 7.3 | Gradio branché sur le LoRA | E7 | A |
| 8.3 | Rapport final + README + ré-exécution sur Colab vierge | E8 | Tous |

---

# Epics et User Stories détaillées

## E1 · Comportement de l'agent

**Étape 1 du sujet — 2 pts · Binôme C**
Définir le rôle de l'agent, ce qu'il traite, ce qu'il refuse, et le contenu des concepts.

### US 1.1 — Identité de l'agent ★ S1
*En tant que correcteur, je veux une description claire de l'agent, pour évaluer si son périmètre est cohérent et assez étroit.*
- **Given** le sujet demande de présenter l'agent
- **When** l'équipe rédige la fiche d'identité
- **Then** elle contient : utilisateur cible, rôle, demandes traitées, demandes refusées, 5 conversations types
- **And** ce texte est réutilisé dans le notebook, dans Gradio (présentation) et dans le rapport

Priorité 10 · Difficulté 3

### US 1.2 — Format des fiches + fiche pilote ★ S1
*En tant qu'équipe, je veux un format de fiche commun, pour que le dataset, les tests et Gradio utilisent les mêmes données.*
- **Given** un concept à couvrir
- **When** on écrit sa fiche
- **Then** elle suit le schéma : `matière`, `domaine`, `concept`, `explication`, `questions[]` avec pour chaque question `type` (définition / exemple / application), `énoncé`, `points_clés[]` (2 à 4)
- **And** une fiche pilote complète existe (ex. photosynthèse, 3 questions)
- **And** le schéma est validé par tous au jour 2

Priorité 10 · Difficulté 3

### US 1.3 — Évaluation en 3 verdicts ★ S1
*En tant qu'utilisateur, je veux une évaluation de chaque réponse, pour savoir ce que j'ai compris et ce qui me manque.*

Regroupe les US Notion existantes : « Mécanisme de l'évaluation », « cas parfait », « cas partielle », « cas inexacte ».

| Verdict | Règle |
|---|---|
| PARFAIT | Tous les points clés présents, aucune erreur |
| PARTIEL | Au moins 1 point correct, d'autres manquants, aucune erreur |
| INCORRECT | Aucun point correct, ou au moins une affirmation fausse |

- **Given** une question et ses points clés
- **When** l'utilisateur donne tous les points sans erreur
- **Then** l'agent écrit `[VERDICT: PARFAIT]`, récapitule et passe à la question suivante
- **When** l'utilisateur donne une partie des points sans erreur
- **Then** l'agent écrit `[VERDICT: PARTIEL]`, confirme les points présents et donne un petit indice par point manquant
- **When** la réponse ne contient aucun point correct ou contient une erreur
- **Then** l'agent écrit `[VERDICT: INCORRECT]`, corrige d'abord l'erreur puis donne de gros indices
- **When** l'utilisateur rate 3 essais sur la même question
- **Then** l'agent montre les points clés attendus et passe à la question suivante
- **And** dans tous les cas, l'agent ne rédige jamais la réponse complète à la place de l'utilisateur

Priorité 10 · Difficulté 5

### US 1.4 — Refus hors périmètre ★ S1
*En tant qu'utilisateur, je veux que l'agent me dise clairement ce qu'il ne fait pas, pour ne pas recevoir de réponse inventée.*

| Cas | Réponse attendue |
|---|---|
| « Donne-moi la réponse » | Refuse, donne un indice à la place |
| Question hors des 3 matières | Redirige vers les matières disponibles |
| Concept absent des fiches | Dit qu'il ne le connaît pas encore, n'improvise pas |
| « Fais mon devoir » | Refuse, propose d'expliquer le concept |
| Réponse vide ou sans rapport | Encourage et reformule la question |

- **Given** une demande de l'un des 5 cas
- **When** l'utilisateur l'envoie
- **Then** l'agent répond selon le tableau, sans verdict et sans inventer de contenu

Priorité 9 · Difficulté 3

### US 1.5 — 6 fiches complètes · S1
*En tant qu'équipe, je veux toutes les fiches prêtes dès la fin du Sprint 1, pour lancer le dataset complet dès le début du Sprint 2.*
- **Given** le format validé (US 1.2)
- **When** l'équipe rédige les fiches
- **Then** 6 fiches existent : 1 concept par domaine (maths, physique, bio, chimie, philosophie, histoire), 3 questions chacune
- **And** chaque point clé est vérifié par une autre personne que l'auteur

Priorité 8 · Difficulté 4

---

## E2 · Observation du modèle

**Étape 2 du sujet — 3 pts · Binôme A**

### US 2.1 — Modèle, tokens, logits, softmax ★ S1
*En tant que correcteur, je veux voir le fonctionnement interne du modèle, pour vérifier que l'équipe comprend la génération.*
- **Given** Colab avec GPU T4
- **When** on charge `Qwen2.5-3B-Instruct` en 4-bit avec `transformers`
- **Then** le notebook affiche, sur un exemple court : les identifiants de tokens, la forme des logits, la distribution du prochain token après softmax (top 10)
- **And** chaque sortie est commentée

Priorité 9 · Difficulté 4

### US 2.2 — Stratégies de décodage + lien avec les TP · S1
*En tant que correcteur, je veux comparer les stratégies de génération, pour vérifier que les paramètres choisis sont justifiés.*
- **Given** le même prompt
- **When** on génère avec argmax, plusieurs températures, top-k et top-p
- **Then** les sorties sont comparées et commentées
- **And** un paragraphe relie ces observations aux embeddings, à l'attention causale et à l'unembedding

Priorité 7 · Difficulté 4

---

## E3 · Baseline conversationnelle

**Étape 3 du sujet — 3 pts · Binôme A**

### US 3.1 — Baseline ★ S1
*En tant qu'équipe, je veux un agent qui fonctionne sans entraînement, pour avoir un point de comparaison avec le LoRA.*
- **Given** la fiche pilote et le modèle de base
- **When** l'utilisateur envoie une réponse
- **Then** le prompt contient un message système (rôle + format + fiche injectée), utilise `apply_chat_template`, et garde un historique limité (N derniers tours)
- **And** la sortie commence par un `[VERDICT: ...]`

Priorité 10 · Difficulté 5

### US 3.2 — Comparaison de prompts · S2
*En tant que correcteur, je veux voir au moins deux configurations comparées, pour juger les choix de prompting.*
- **Given** les tests de l'US 6.1
- **When** on compare 2 prompts (ou zero-shot contre few-shot)
- **Then** les résultats sont notés avec la grille
- **And** les paramètres de génération sont justifiés
- **And** les réponses de la meilleure configuration sont sauvegardées comme référence

Priorité 8 · Difficulté 4

---

## E4 · Dataset de dialogues

**Étape 4 du sujet — 2 pts · Binôme B (+ relecture par tous)**

### US 4.1 — Format, script, 30 dialogues pilotes ★ S1
*En tant qu'équipe, je veux un format de dialogue et un script reproductible, pour produire le dataset rapidement et le livrer.*
- **Given** la fiche pilote
- **When** le script génère des dialogues
- **Then** chaque dialogue est au format JSONL `{"messages": [system, user, assistant, ...]}`
- **And** 30 dialogues pilotes existent, relus à la main, couvrant les 3 verdicts et au moins 3 refus
- **And** le format est validé par tous au jour 2

Priorité 10 · Difficulté 5

### US 4.2 — Dataset complet · S2
*En tant qu'équipe, je veux un dataset propre et équilibré, pour entraîner un adaptateur fiable.*
- **Given** les 6 fiches (US 1.5)
- **When** le dataset est généré puis relu
- **Then** il contient environ 300 dialogues, équilibrés entre matières, verdicts (≈ 30 % parfait, 35 % partiel, 35 % incorrect) et refus (≈ 15 %)
- **And** chaque dialogue est relu par un humain, sans donnée personnelle
- **And** il est séparé en train / validation / test (ex. 80 / 10 / 10)
- **And** aucune requête des 20 tests (US 6.2) n'est dans le train

Priorité 10 · Difficulté 7

---

## E5 · Adaptation LoRA / QLoRA

**Étape 5 du sujet — 4 pts · Binôme B**

### US 5.1 — Pipeline QLoRA minimal ★ S1
*En tant qu'équipe, je veux un entraînement qui tourne dès la semaine 1, pour découvrir tôt les problèmes de Colab (mémoire, format).*
- **Given** les 30 dialogues pilotes
- **When** on lance un entraînement QLoRA court (≈ 5 min)
- **Then** le modèle de base reste gelé, la loss s'affiche, l'adaptateur est sauvegardé et rechargeable
- **And** la durée et la mémoire utilisées sont notées (pour planifier la US 5.2)

Priorité 10 · Difficulté 7

### US 5.2 — Entraînement complet · S2
*En tant que correcteur, je veux voir un entraînement suivi et justifié, pour évaluer l'adaptation.*
- **Given** le dataset complet (US 4.2)
- **When** on entraîne l'adaptateur
- **Then** les courbes de loss train et validation sont affichées
- **And** le meilleur adaptateur selon la validation est sauvegardé
- **And** l'arrêt de l'entraînement est justifié (ex. la loss de validation remonte)

Priorité 10 · Difficulté 8

### US 5.3 — Documentation + publication · S3
*En tant que correcteur, je veux tous les réglages et l'adaptateur, pour reproduire le résultat.*
- **Given** l'adaptateur final
- **When** on prépare le rendu
- **Then** sont documentés : modèle de base, quantification, rang LoRA, taux d'apprentissage, taille des lots, nombre d'époques, longueur maximale
- **And** l'adaptateur est livré ou accessible par un lien stable

Priorité 9 · Difficulté 3

---

## E6 · Évaluation avant / après

**Étape 6 du sujet — 3 pts · Binôme C**

### US 6.1 — Grille + 5 premiers tests ★ S1
*En tant qu'équipe, je veux une grille définie avant tout entraînement, pour que la comparaison soit honnête.*
- **Given** les critères du sujet
- **When** on rédige la grille
- **Then** elle couvre : pertinence et correction · respect du rôle et du format · prise en compte de l'historique · reconnaissance du hors périmètre · informations inventées
- **And** chaque critère a une échelle explicite (ex. 0 / 1 / 2)
- **And** 5 tests existent (dont 1 refus) et la baseline est notée

Priorité 10 · Difficulté 3

### US 6.2 — 20 tests + score automatique · S2
*En tant que correcteur, je veux au moins 20 tests jamais vus à l'entraînement, pour mesurer l'effet du LoRA.*
- **Given** les 6 fiches
- **When** on écrit les tests **avant** l'entraînement complet (US 5.2)
- **Then** il y a au moins 20 requêtes, dont au moins 3 conversations sur plusieurs tours (historique) et 5 refus
- **And** chaque test a son verdict attendu
- **And** un score automatique compare le verdict prédit au verdict attendu

Priorité 10 · Difficulté 4

### US 6.3 — Comparaison et analyse · S3
*En tant que correcteur, je veux un tableau de comparaison et une analyse des erreurs, pour juger ce que le LoRA a apporté.*
- **Given** la baseline (E3) et le modèle adapté (E5)
- **When** on passe les 20 tests aux deux versions
- **Then** un tableau synthétique compare les scores par critère
- **And** au moins 5 réussites ou erreurs sont commentées

Priorité 10 · Difficulté 5

---

## E7 · Interface Gradio

**Étape 7 du sujet — partie des 3 pts « interface, reproductibilité, rendu » · Binôme A**

### US 7.1 — Gradio v0.1 ★ S1
*En tant qu'utilisateur, je veux une interface de chat simple, pour parler à l'agent.*
- **Given** le notebook lancé
- **When** on ouvre l'interface
- **Then** elle contient une zone de conversation, un champ de saisie, un bouton de réinitialisation, et une courte présentation du rôle et des limites (texte de l'US 1.1)
- **And** elle est branchée sur la baseline

Priorité 9 · Difficulté 3

### US 7.2 — Parcours complet · S2
*En tant qu'utilisateur, je veux choisir une matière et enchaîner les questions, pour apprendre un concept du début à la fin.*
- **Given** les 6 fiches
- **When** je choisis une matière
- **Then** l'interface affiche le concept puis pose les questions dans l'ordre (définition → exemple → application)
- **And** elle lit le `[VERDICT]` : PARFAIT → question suivante ; sinon nouvel essai ; au 3e échec → points clés affichés puis question suivante
- **And** après la dernière question, elle propose un autre concept

Priorité 8 · Difficulté 5

### US 7.3 — Branchement sur le LoRA · S3
*En tant qu'utilisateur, je veux parler au modèle adapté, pour profiter de la meilleure version de l'agent.*
- **Given** l'adaptateur final (US 5.2)
- **When** on lance l'interface
- **Then** elle utilise le modèle adapté
- **And** (optionnel) un sélecteur permet de comparer avec la baseline

Priorité 8 · Difficulté 2

---

## E8 · Livrables et rapport

**Livrables du sujet — partie des 3 pts « interface, reproductibilité, rendu » · Tous**

### US 8.1 — Squelette du notebook + règles ★ S1
*En tant qu'équipe, je veux une structure commune dès le début, pour travailler à 6 sans se marcher dessus.*
- **Given** 3 binômes travaillant en parallèle
- **When** le projet démarre
- **Then** `agent.ipynb` contient les sections 1 à 7 du sujet (vides au départ)
- **And** un dossier Drive partagé existe (fiches, dataset, adaptateurs)
- **And** règle : chaque binôme travaille dans son propre sous-notebook ; assemblage dans `agent.ipynb` chaque vendredi

Priorité 10 · Difficulté 2

### US 8.2 — Rapport partie 1 · S2
*En tant qu'équipe, je veux commencer le rapport tôt, pour ne pas tout écrire la dernière semaine.*
- **Given** E1 et E4 terminées
- **When** on rédige
- **Then** les sections « cas d'usage » et « données » sont écrites

Priorité 7 · Difficulté 3

### US 8.3 — Rendu final · S3
*En tant que correcteur, je veux des livrables complets et reproductibles, pour exécuter et noter le projet.*
- **Given** toutes les epics terminées
- **When** on prépare le rendu
- **Then** `rapport.pdf` fait 4 pages maximum hors annexes : cas d'usage, données, choix techniques, comparaison des deux versions, limites, améliorations (dont le RAG)
- **And** `README.txt` donne les instructions d'exécution
- **And** `agent.ipynb` est ré-exécuté de bout en bout sur un Colab vierge, sorties conservées
- **And** les données (ou le script) et l'adaptateur (ou le lien) sont livrés

Priorité 10 · Difficulté 4

---

# À valider par l'équipe

Chacun coche ou commente. Si une ligne pose problème, on en parle avant de toucher à Notion.

## Structure

- [ ] Les 8 epics (E1 à E8), calquées sur les étapes du sujet
- [ ] Les 23 US et leur répartition S1 / S2 / S3
- [ ] Sprint 1 = MVP de bout en bout (1 concept, tout le pipeline), puis passage à l'échelle
- [ ] Répartition des binômes : A = E2 + E3 + E7 · B = E4 + E5 · C = E1 + E6 · tous = E8 + relecture dataset → **qui va dans quel binôme ?**
- [ ] Les US Notion existantes (« Mécanisme de l'évaluation », « cas parfait », « cas partielle », « cas inexacte ») deviennent les critères de la US 1.3

## Contenu

- [ ] 3 verdicts + tag `[VERDICT]` + 3 essais max (US 1.3)
- [ ] Les 5 cas de refus (US 1.4)
- [ ] 3 questions par concept : définition → exemple → application
- [ ] Fiches JSON dans le prompt, pas de RAG (RAG cité en amélioration dans le rapport)
- [ ] Modèle `Qwen2.5-3B-Instruct` sur Colab

## Précisions ajoutées sans discussion (à confirmer)

- [ ] Répartition du dataset : ≈ 30 % parfait, 35 % partiel, 35 % incorrect, dont ≈ 15 % de refus ; split 80 / 10 / 10 (US 4.2)
- [ ] Un sous-notebook par binôme, assemblage dans `agent.ipynb` chaque vendredi (US 8.1)
- [ ] Une US est validée par quelqu'un d'autre que son auteur

## Questions sans réponse

1. **Nombre de concepts** : 1 par domaine (6 concepts, recommandé pour tenir en 3 semaines) ou 2 par domaine (12) ?
2. **Génération des dialogues avec un gros LLM externe** (ChatGPT, Claude) : autorisée ? → **à demander au prof dès le jour 1.** En attendant, on prépare le format et les gabarits (US 4.1), utiles dans les deux cas. Si c'est interdit : dialogues écrits à la main avec gabarits, viser ~200.
3. **Usage de l'IA pour coder** : autorisé. Faut-il le déclarer dans le rapport ? → à demander au prof en même temps que la question 2.
4. **Notion** : garder la base « Sprint | 1 » telle quelle, ou la renommer « User Stories » avec une propriété « Sprint » (S1 / S2 / S3) et une vue par sprint ?
