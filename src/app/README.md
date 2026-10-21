# Interface Brainmaxxing (Gradio)

L'interface est définie par `src/app/gradio_app.py`. Le thème clair, les
couleurs des fiches, le squelette de chargement et les fragments HTML sont dans
`src/app/gradio_theme.py`.

L'accueil présente les fiches sous forme de cartes. Une carte ouvre la page de
lecture : le cours s'y affiche au fil des questions, et la console du tuteur,
en bas de l'écran, reçoit les réponses. La flèche en haut à gauche ramène à
l'accueil ; l'interrupteur B/Q de la console choisit `baseline` ou `lora`.

## Installation et lancement

Depuis la racine du dépôt :

```powershell
python -m pip install -r requirements-gradio.txt
python -m src.app.gradio_app
```

En notebook :

```python
from src.app.gradio_app import launch_app
launch_app(share=False)
```

Utiliser `launch_app()` (et non `build_app().launch()`) pour appliquer le thème
avec Gradio 6 : la bibliothèque applique désormais `theme` et `css` au moment
de `launch()`. Avec Gradio 5, ces paramètres restent sur `Blocks`.

L'interface conserve le fonctionnement du moteur : fiches progressives,
questions, indices, trois tentatives et historique pédagogique. Le mode `lora` nécessite l'adaptateur QLoRA
v1.1 et les dépendances décrites dans le README principal du dépôt.

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

Les tests de présentation vérifient la progression et le montage de
l'interface sans télécharger Qwen. Une validation manuelle de l'affichage en
navigateur et du modèle QLoRA sur GPU reste nécessaire.
