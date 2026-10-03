# Règles de travail avec les agents IA

Ces règles s'appliquent à tout agent IA utilisé sur ce dépôt : Claude Code, Cursor, Copilot, Codex, ChatGPT, etc. Elles s'appliquent aussi aux personnes qui collent du code produit par une IA.

Objectif : chaque membre de l'équipe garde le contrôle du code et comprend tout ce qu'il commit.

## 1. Une étape = un petit morceau

- Une étape produit **une seule fonction** ou **une seule cellule** de notebook. Environ 40 lignes au maximum.
- Une étape touche **un seul fichier**.
- À la fin de l'étape, l'agent s'arrête et attend la validation avant de continuer.
- Si la demande dépasse ce cadre, l'agent propose un découpage en étapes au lieu de tout coder d'un coup.

## 2. Le plan avant le code

Avant d'écrire du code, l'agent annonce en 3 lignes maximum :

1. la fonction qu'il va écrire (nom et rôle) ;
2. ses entrées et ses sorties ;
3. le contrat concerné, s'il y en a un (voir règle 6).

Il attend un « ok » avant de coder.

## 3. Commentaires

- **Langue** : commentaires et docstrings en français. Noms de fonctions et de variables en anglais, en `snake_case`. Les clés JSON des contrats restent telles qu'elles sont définies (`matiere`, `points_cles`, etc.).
- **Docstring obligatoire** pour chaque fonction :

```python
def parse_response(raw: str) -> dict:
    """Transforme la sortie brute du modèle en dictionnaire exploitable.

    Paramètres :
        raw : texte généré par le modèle.

    Retour :
        {"verdict", "feedback", "raw_response", "valid_format"} (contrat 2).

    Exemple :
        parse_response("[VERDICT: PARFAIT]\nBravo.")["verdict"]  # "PARFAIT"
    """
```

- **Commentaires en ligne** : ils expliquent le *pourquoi*, pas le *quoi*. Pas de commentaire qui répète le code.

```python
# Bien : explique un choix
lines = raw.strip().splitlines()  # le modèle ajoute parfois des lignes vides au début

# À éviter : répète le code
lines = raw.strip().splitlines()  # découpe le texte en lignes
```

## 4. Compréhension obligatoire

- Après chaque étape, l'agent explique le code en 3 à 5 points simples.
- Sur demande, il explique une ligne précise.
- **Règle pour la personne qui code : si tu ne peux pas expliquer une ligne, tu ne la commits pas.** Demande une explication ou une version plus simple.
- L'agent préfère toujours la solution la plus simple et la plus lisible à la solution la plus courte ou la plus élégante.

## 5. Un test par fonction

- Chaque fonction est livrée avec au moins un test simple à base d'`assert`, écrit dans la même étape.
- Le test est exécuté avant la validation. L'agent montre le résultat réel de l'exécution.

```python
assert parse_response("[VERDICT: PARTIEL]\nPresque.")["verdict"] == "PARTIEL"
assert parse_response("texte sans tag")["valid_format"] is False
```

## 6. Les contrats sont intouchables

Les 4 contrats communs sont décrits dans [backlog.md](backlog.md#contrats-communs) et sur Notion :

1. fiche concept ;
2. sortie de l'agent ;
3. fonction d'inférence `generate_response` ;
4. cas de test.

- Un agent **ne modifie jamais** le format d'un contrat.
- Si un contrat semble incomplet ou faux, l'agent le signale. La personne qui code en parle à l'équipe avant tout changement.

## 7. Rester dans son périmètre

- Chacun travaille dans le code de son bloc (voir la section Équipe de [backlog.md](backlog.md#équipe)).
- Pas de modification du code d'un autre bloc sans son accord.
- Pas de refactor non demandé, pas de fonctionnalité en plus, pas d'abstraction « pour plus tard ».
- Pas de nouvelle dépendance sans accord de l'équipe. Les versions des bibliothèques sont figées (`pip install paquet==x.y.z`) pour que le notebook tourne pareil sur Colab.

## 8. Git

- Une branche par bloc (exemples : `bloc-a-pedagogie`, `bloc-e-gradio`).
- Un commit par fonction ou par petite étape. Format du message :
  - `feat: ajoute parse_response`
  - `fix: gère la sortie vide dans parse_response`
  - `docs: complète le README`
  - `test: ajoute les cas hors périmètre`
- Merge sur `main` seulement après relecture par une autre personne que l'auteur (Definition of Done).
- Pendant le développement, les sorties des notebooks sont vidées avant chaque commit. Les sorties ne sont conservées que pour le rendu final.

## 9. README neutre et informatif

Le README décrit le projet tel qu'il est, pas comment il a été fait. Il contient :

- ce que fait le projet ;
- l'installation et l'exécution (Colab) ;
- la structure des dossiers ;
- un lien vers les contrats.

Le README ne contient pas : d'historique, de « nous avons… », de journal des modifications, de récit du développement, de mention des outils d'IA utilisés.

## 10. Sécurité et honnêteté

- Aucun secret (token Hugging Face, clé d'API) dans le code ou le notebook. Utiliser les secrets Colab (`from google.colab import userdata`).
- L'agent n'invente jamais de résultat : loss, score, temps d'entraînement. Seuls les chiffres produits par une exécution réelle sont écrits dans le code, le rapport ou le README.
- Si l'agent n'est pas sûr d'une API ou d'un paramètre, il le dit au lieu de deviner.
