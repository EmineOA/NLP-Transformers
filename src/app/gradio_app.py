"""Interface Gradio de l'agent pédagogique (bloc E7).

L'interface ne dépend que de `generate_response` (contrat 3) et des fiches
concept (contrat 1). Le parcours (essais, question suivante) est géré ici,
jamais par le modèle.
"""
from __future__ import annotations

from typing import Callable

from src.agents.concepts import CONCEPT_FOLDER, load_concept
from src.agents.pedagogical_context import build_student_message, summarize_history
from src.app.gradio_theme import CUSTOM_CSS, FOOTER_HTML, HEADER_HTML, SIDEBAR_GUIDE_HTML, make_theme

MAX_ATTEMPTS = 3

VERDICT_LABELS = {
    "PARFAIT": "✅ Parfait",
    "PARTIEL": "🟡 Partiel",
    "INCORRECT": "❌ Incorrect",
}
OUT_OF_SCOPE_LABEL = "⛔ Demande refusée"

PRESENTATION = (
    "### Agent pédagogique\n"
     "Choisis un concept, découvre progressivement son cours et réponds aux questions "
    "de la fiche. L'agent évalue ta réponse : "
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
    return {"concept_id": None, "question_index": 0, "attempts": 0,
            "messages": [], "finished": False, "shown_sections": 0,
            "completed_answers": [], "question_attempts": []}


def _question_text(concept: dict, state: dict) -> str:
    """Texte de la question courante, numérotée."""
    questions = concept["questions"]
    index = state["question_index"]
    return f"**Question {index + 1}/{len(questions)}** : {questions[index]['enonce']}"


def _next_course_sections(concept: dict, state: dict) -> str:
    """Révèle seulement les nouvelles sections autorisées pour la question."""
    until = concept["questions"][state["question_index"]]["apres_section"]
    previous = state["shown_sections"]
    state["shown_sections"] = until
    return "\n\n".join(
        f"### {section['titre']}\n{section['contenu']}"
        for section in concept["explication"][previous:until]
    )


def start_concept(concept_id: str) -> tuple[dict, list[dict]]:
    """Charge un concept et prépare le début du parcours.

    Paramètres :
        concept_id : identifiant de la fiche choisie.

    Retour :
        (état initial, historique du chat avec l'explication et la 1re question).

    Exemple :
        start_concept("monty-hall")[0]["question_index"]  # 0
    """
    if not concept_id:
        return reset_session()
    concept = load_concept(concept_id)
    state = _new_state()
    state["concept_id"] = concept_id
    explanation = _next_course_sections(concept, state)
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
            hint = question["indices"][state["attempts"] - 1]
            return state, f"**Indice {state['attempts']}** : {hint}\n\nTu peux réessayer."
        points = "\n".join(f"- {p}" for p in question["points_cles"])
        note = f"Voici ce qui était attendu :\n{points}\n\n"

    if state.get("last_answer"):
        state["completed_answers"] = state["completed_answers"] + [
            {"question_id": question["id"], "response": state["last_answer"]}
        ]
    state["attempts"] = 0
    state["question_attempts"] = []
    state["question_index"] += 1
    if state["question_index"] >= len(concept["questions"]):
        state["finished"] = True
        return state, note + "Concept terminé. Choisis un autre concept dans la liste pour continuer."
    sections = _next_course_sections(concept, state)
    if sections:
        note += sections + "\n\n"
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
    answer = user_text.strip()
    state["last_answer"] = answer
    chat.append({"role": "user", "content": answer})

    concept = load_concept(state["concept_id"])
    question = concept["questions"][state["question_index"]]
    history_summary = summarize_history(
        state["completed_answers"], state["question_attempts"], question["id"],
    )
    input_text = build_student_message(
        answer, history=history_summary,
        hints_seen=question["indices"][:state["attempts"]],
    )
    # Comme dans le dataset : un message utilisateur structuré, sans cible supervisée.
    try:
        result = generate([{"role": "user", "content": input_text}], concept, question, model_version)
    except Exception as exc:  # l'interface doit continuer à fonctionner quoi qu'il arrive
        result = {"verdict": None, "feedback": f"Erreur de l'agent : {exc}", "raw_response": "", "valid_format": False}

    state["messages"] = state["messages"] + [
        {"role": "user", "content": answer},
        {"role": "assistant", "content": result.get("raw_response") or result.get("feedback") or ""},
    ]
    chat.append({"role": "assistant", "content": format_verdict(result)})
    if result.get("valid_format") and result.get("verdict") in VERDICT_LABELS:
        state["question_attempts"] = state["question_attempts"] + [
            {"response": answer, "verdict": result["verdict"]}
        ]
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


def render_progress(state: dict) -> str:
    """Résumé du parcours affiché dans l'interface, sans modifier l'état."""
    if not state.get("concept_id"):
        label, detail, percent = "Prêt à commencer", "Choisis un concept pour démarrer.", 0
    else:
        concept = load_concept(state["concept_id"])
        total = len(concept["questions"])
        if state.get("finished"):
            label, detail, percent = "Concept terminé", f"{total} question(s) parcourue(s).", 100
        else:
            index = state["question_index"]
            attempt = min(state["attempts"] + 1, MAX_ATTEMPTS)
            label = f"Question {index + 1} / {total}"
            detail = f"Tentative {attempt} / {MAX_ATTEMPTS}"
            percent = round(100 * index / total)
    return (
        '<div class="bm-progress-card">'
        '<div class="bm-progress-head"><span class="bm-progress-title">' + label + '</span></div>'
        '<div class="bm-progress-meta">' + detail + '</div>'
        '<div class="bm-progress-track" role="progressbar" '
        f'aria-label="Progression du concept" aria-valuemin="0" aria-valuemax="100" '
        f'aria-valuenow="{percent}"><span class="bm-progress-fill" style="width:{percent}%"></span></div>'
        '</div>'
    )


def build_app():
    """Construit la vue Gradio. Le parcours et les appels au LLM restent inchangés."""
    import gradio as gr

    with gr.Blocks(
        title="Brainmaxxing — Tuteur pédagogique",
        theme=make_theme(gr),
        css=CUSTOM_CSS,
    ) as app:
        state = gr.State(_new_state())
        gr.HTML(HEADER_HTML)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=285):
                with gr.Group(elem_classes=["bm-panel"]):
                    gr.Markdown("### Ton espace", elem_classes=["bm-side-title"])
                    gr.Markdown(
                        "Choisis un sujet et une configuration pour commencer.",
                        elem_classes=["bm-side-intro"],
                    )
                    concept_choice = gr.Dropdown(
                        choices=list_concepts(),
                        label="Concept à découvrir",
                        info="Un cours guidé, question après question",
                        value=None,
                        interactive=True,
                        elem_classes=["bm-field"],
                    )
                    model_choice = gr.Radio(
                        choices=[
                            ("Qwen · Baseline", "baseline"),
                            ("Qwen · QLoRA", "lora"),
                        ],
                        value="baseline",
                        label="Version du tuteur",
                        elem_classes=["bm-field"],
                    )
                    reset = gr.Button(
                        "↻ Recommencer",
                        variant="secondary",
                        elem_classes=["bm-action-row"],
                    )

                with gr.Group(elem_classes=["bm-panel"]):
                    gr.Markdown("### Comment ça marche ?", elem_classes=["bm-side-title"])
                    gr.HTML(SIDEBAR_GUIDE_HTML)
                    with gr.Accordion("Rôle et limites du tuteur", open=False):
                        gr.Markdown(
                            "Le tuteur évalue le raisonnement avec la fiche du concept. "
                            "Il peut faire des erreurs et ne remplace pas un enseignant. "
                            "Trois tentatives sont possibles par question, avec deux indices."
                        )

            with gr.Column(scale=2, min_width=400):
                with gr.Group(elem_classes=["bm-panel"]):
                    gr.Markdown("### Ton dialogue", elem_classes=["bm-main-title"])
                    gr.Markdown(
                        "Réponds naturellement. Un indice apparaît si ta réponse est incomplète.",
                        elem_classes=["bm-main-subtitle"],
                    )
                    progress = gr.HTML(render_progress(_new_state()))
                    chatbot = gr.Chatbot(
                        type="messages",
                        height=540,
                        label="Échanges avec le tuteur",
                        show_copy_button=True,
                        elem_classes=["bm-chat"],
                    )
                    textbox = gr.Textbox(
                        placeholder="Quel est ton raisonnement ? Explique ton choix…",
                        label="Ta réponse",
                        lines=3,
                        max_lines=6,
                        elem_classes=["bm-input"],
                    )
                    send = gr.Button(
                        "Envoyer ma réponse →",
                        variant="primary",
                        elem_classes=["bm-primary"],
                    )

        gr.HTML(FOOTER_HTML)

        def on_concept_change(concept_id):
            new_state, new_chat = start_concept(concept_id)
            return new_state, new_chat, render_progress(new_state), ""

        def on_send(text, current_state, history, version):
            new_state, new_chat = submit_answer(text, current_state, history, version)
            return new_state, new_chat, "", render_progress(new_state)

        def on_reset():
            new_state, new_chat = reset_session()
            return new_state, new_chat, None, "", render_progress(new_state)

        concept_choice.change(
            on_concept_change, concept_choice, [state, chatbot, progress, textbox]
        )
        send.click(
            on_send, [textbox, state, chatbot, model_choice],
            [state, chatbot, textbox, progress],
        )
        textbox.submit(
            on_send, [textbox, state, chatbot, model_choice],
            [state, chatbot, textbox, progress],
        )
        reset.click(
            on_reset, None, [state, chatbot, concept_choice, textbox, progress]
        )
    return app


if __name__ == "__main__":
    build_app().launch()
