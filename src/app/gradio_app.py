"""Interface Gradio de l'agent pédagogique (bloc E7).

L'interface ne dépend que de `generate_response` (contrat 3) et des fiches
concept (contrat 1). Le parcours (essais, question suivante) est géré ici,
jamais par le modèle.
"""
from __future__ import annotations

from typing import Callable

from src.agents.concepts import CONCEPT_FOLDER, load_concept

MAX_ATTEMPTS = 3

VERDICT_LABELS = {
    "PARFAIT": "✅ Parfait",
    "PARTIEL": "🟡 Partiel",
    "INCORRECT": "❌ Incorrect",
}
OUT_OF_SCOPE_LABEL = "⛔ Hors sujet"

PRESENTATION = (
    "### Agent pédagogique\n"
    "Choisis un concept, lis son explication, puis réponds aux 3 questions "
    "(définition, exemple, application). L'agent évalue ta réponse : "
    "✅ Parfait, 🟡 Partiel, ❌ Incorrect ou ⛔ Hors sujet.\n\n"
    "**Limites** : il ne connaît que les concepts de la liste, tu as "
    f"{MAX_ATTEMPTS} essais par question, et il peut se tromper."
)


def list_concepts() -> list[tuple[str, str]]:
    """Liste les concepts disponibles pour la liste déroulante.

    Retour :
        [(libellé affiché, identifiant)], trié par libellé.

    Exemple :
        ("Monty Hall", "monty-hall") in list_concepts()
    """
    choices = []
    for path in sorted(CONCEPT_FOLDER.glob("*.json")):
        concept = load_concept(path.stem)
        choices.append((concept["concept"], concept["id"]))
    return sorted(choices)


def _new_state() -> dict:
    """État vide : aucun concept choisi."""
    return {"concept_id": None, "question_index": 0, "attempts": 0, "messages": [], "finished": False}


def _question_text(concept: dict, state: dict) -> str:
    """Texte de la question courante, numérotée."""
    questions = concept["questions"]
    index = state["question_index"]
    return f"**Question {index + 1}/{len(questions)}** : {questions[index]['enonce']}"


def start_concept(concept_id: str) -> tuple[dict, list[dict]]:
    """Charge un concept et prépare le début du parcours.

    Paramètres :
        concept_id : identifiant de la fiche choisie.

    Retour :
        (état initial, historique du chat avec l'explication et la 1re question).

    Exemple :
        start_concept("monty-hall")[0]["question_index"]  # 0
    """
    concept = load_concept(concept_id)
    state = _new_state()
    state["concept_id"] = concept_id
    explanation = "\n\n".join(f"### {s['titre']}\n{s['contenu']}" for s in concept["explication"])
    chat = [
        {"role": "assistant", "content": f"## {concept['concept']}\n\n{explanation}"},
        {"role": "assistant", "content": _question_text(concept, state)},
    ]
    return state, chat


def format_verdict(result: dict) -> str:
    """Met le verdict en évidence devant le feedback de l'agent.

    Paramètres :
        result : dictionnaire du contrat 2 (résultat de generate_response).

    Retour :
        texte à afficher dans le chat.

    Exemple :
        format_verdict({"verdict": "PARFAIT", "feedback": "Bravo.", "valid_format": True})
        # "**✅ Parfait**\n\nBravo."
    """
    if not result.get("valid_format"):
        # sortie mal formée : affichée telle quelle, sans verdict inventé
        return str(result.get("raw_response") or result.get("feedback") or "")
    label = VERDICT_LABELS.get(result.get("verdict"), OUT_OF_SCOPE_LABEL)
    return f"**{label}**\n\n{result.get('feedback', '')}"


def next_step(state: dict, result: dict) -> tuple[dict, str]:
    """Décide la suite du parcours à partir du verdict et du compteur d'essais.

    Paramètres :
        state : état courant (question, essais, messages).
        result : dictionnaire du contrat 2.

    Retour :
        (nouvel état, message d'accompagnement à afficher, éventuellement vide).

    Exemple :
        next_step(state, {"verdict": "PARFAIT", "valid_format": True})[0]["question_index"]
    """
    state = dict(state)
    verdict = result.get("verdict")
    if not result.get("valid_format") or verdict is None:
        # sortie invalide ou hors périmètre : le parcours n'avance pas et l'essai n'est pas compté
        return state, ""

    concept = load_concept(state["concept_id"])
    question = concept["questions"][state["question_index"]]
    note = ""
    if verdict != "PARFAIT":
        state["attempts"] += 1
        if state["attempts"] < MAX_ATTEMPTS:
            return state, f"Essai {state['attempts']}/{MAX_ATTEMPTS} : tu peux réessayer."
        points = "\n".join(f"- {p}" for p in question["points_cles"])
        note = f"Voici ce qui était attendu :\n{points}\n\n"

    state["attempts"] = 0
    state["question_index"] += 1
    if state["question_index"] >= len(concept["questions"]):
        state["finished"] = True
        return state, note + "Concept terminé. Choisis un autre concept dans la liste pour continuer."
    return state, note + _question_text(concept, state)


def submit_answer(
    user_text: str,
    state: dict,
    chat: list[dict],
    model_version: str = "baseline",
    generate: Callable | None = None,
) -> tuple[dict, list[dict]]:
    """Évalue la réponse de l'étudiant et met à jour l'état et le chat.

    Paramètres :
        user_text : réponse saisie.
        state : état courant.
        chat : historique affiché.
        model_version : 'baseline' ou 'lora'.
        generate : fonction d'inférence (contrat 3), injectable pour les tests.

    Retour :
        (nouvel état, nouvel historique du chat).

    Exemple :
        submit_answer("ma réponse", state, chat, generate=faux_generate)
    """
    chat = list(chat)
    if not user_text or not user_text.strip():
        return state, chat
    if state["concept_id"] is None or state["finished"]:
        chat.append({"role": "assistant", "content": "Choisis d'abord un concept dans la liste."})
        return state, chat
    if generate is None:
        # import tardif : charger le modèle seulement quand on en a besoin
        from src.agents.inference import generate_response as generate

    state = dict(state)
    state["messages"] = state["messages"] + [{"role": "user", "content": user_text.strip()}]
    chat.append({"role": "user", "content": user_text.strip()})

    concept = load_concept(state["concept_id"])
    question = concept["questions"][state["question_index"]]
    try:
        result = generate(state["messages"], concept, question, model_version)
    except Exception as exc:  # l'interface doit continuer à fonctionner quoi qu'il arrive
        result = {"verdict": None, "feedback": f"Erreur de l'agent : {exc}", "raw_response": "", "valid_format": False}

    state["messages"].append({"role": "assistant", "content": result.get("raw_response") or result.get("feedback") or ""})
    chat.append({"role": "assistant", "content": format_verdict(result)})
    state, note = next_step(state, result)
    if note:
        chat.append({"role": "assistant", "content": note})
    return state, chat


def reset_session() -> tuple[dict, list[dict]]:
    """Remet l'interface à zéro (aucun concept, chat vide).

    Exemple :
        reset_session()[0]["concept_id"] is None
    """
    return _new_state(), []


def build_app():
    """Assemble l'interface Gradio (chat, saisie, choix du concept, reset).

    Retour :
        l'objet gr.Blocks, à lancer avec .launch().
    """
    import gradio as gr  # import tardif : les tests de logique n'en ont pas besoin

    with gr.Blocks(title="Agent pédagogique") as app:
        gr.Markdown(PRESENTATION)
        state = gr.State(_new_state())
        with gr.Row():
            concept_choice = gr.Dropdown(choices=list_concepts(), label="Concept")
            # sélecteur de démo : à brancher sur l'adaptateur final (T7.1)
            model_choice = gr.Radio(["baseline", "lora"], value="baseline", label="Modèle")
        chatbot = gr.Chatbot(type="messages", height=480, label="Conversation")
        textbox = gr.Textbox(placeholder="Écris ta réponse…", label="Ta réponse")
        with gr.Row():
            send = gr.Button("Envoyer", variant="primary")
            reset = gr.Button("Recommencer à zéro")

        def on_send(text, current_state, history, version):
            new_state, new_chat = submit_answer(text, current_state, history, version)
            return new_state, new_chat, ""

        concept_choice.change(start_concept, concept_choice, [state, chatbot])
        send.click(on_send, [textbox, state, chatbot, model_choice], [state, chatbot, textbox])
        textbox.submit(on_send, [textbox, state, chatbot, model_choice], [state, chatbot, textbox])
        reset.click(reset_session, None, [state, chatbot])
    return app


if __name__ == "__main__":
    build_app().launch()
