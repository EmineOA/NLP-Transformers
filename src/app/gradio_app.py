"""Interface Gradio de l'agent pédagogique (bloc E7).

L'interface ne dépend que de `generate_response` (contrat 3) et des fiches
concept (contrat 1). Le parcours (essais, question suivante) est géré ici,
jamais par le modèle.
"""
from __future__ import annotations

import inspect
from functools import partial
from typing import Callable

from src.agents.concepts import CONCEPT_FOLDER, load_concept
from src.agents.pedagogical_context import build_student_message, summarize_history
from src.app.gradio_theme import (
    CONCEPT_STYLES, CUSTOM_CSS, HEAD_HTML, HEADER_HTML, LOGO_HTML, card_html, console_title_html,
    make_theme, story_header_html, switch_html,
)

MAX_ATTEMPTS = 3

VERDICT_LABELS = {
    "PARFAIT": "✅ Parfait",
    "PARTIEL": "🟡 Partiel",
    "INCORRECT": "❌ Incorrect",
}
OUT_OF_SCOPE_LABEL = "⛔ Demande refusée"


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


def _next_course_sections(concept: dict, state: dict) -> list[dict]:
    """Révèle seulement les nouvelles sections autorisées pour la question.

    Retour :
        liste des sections nouvellement visibles (vide si rien de nouveau).
    """
    until = concept["questions"][state["question_index"]]["apres_section"]
    previous = state["shown_sections"]
    state["shown_sections"] = max(previous, until)
    return concept["explication"][previous:until]


def render_course(state: dict) -> str:
    """Texte du cours visible dans la page de lecture (Markdown).

    Paramètres :
        state : état courant ; seules les `shown_sections` premières sections sont rendues.

    Retour :
        Markdown des sections déjà débloquées, vide si aucun concept.

    Exemple :
        render_course(start_concept("monty-hall")[0]).startswith("## Introduction")  # True
    """
    if not state.get("concept_id"):
        return ""
    concept = load_concept(state["concept_id"])
    return "\n\n".join(
        f"## {section['titre']}\n\n{section['contenu']}"
        for section in concept["explication"][:state["shown_sections"]]
    )


def start_concept(concept_id: str) -> tuple[dict, list[dict]]:
    """Charge un concept et prépare le début du parcours.

    Paramètres :
        concept_id : identifiant de la fiche choisie.

    Retour :
        (état initial, historique du chat avec la 1re question).
        Le cours lui-même est affiché à part, voir `render_course`.

    Exemple :
        start_concept("monty-hall")[0]["question_index"]  # 0
    """
    if not concept_id:
        return reset_session()
    concept = load_concept(concept_id)
    state = _new_state()
    state["concept_id"] = concept_id
    _next_course_sections(concept, state)
    chat = [{"role": "assistant", "content": _question_text(concept, state)}]
    return state, chat


def format_verdict(result: dict) -> str:
    """Met le verdict en évidence devant le feedback de l'agent.

    Paramètres :
        result : dictionnaire du contrat 2 (résultat de generate_response).

    Retour :
        texte à afficher dans le chat.

    Exemple :
        format_verdict({"verdict": "PARFAIT", "feedback": "Bravo.", "valid_format": True})
        # '<span class="v PARFAIT">✅ Parfait</span>\n\nBravo.'
    """
    if not result.get("valid_format"):
        # sortie mal formée : affichée telle quelle, sans verdict inventé
        return str(result.get("raw_response") or result.get("feedback") or "")
    verdict = result.get("verdict")
    label = VERDICT_LABELS.get(verdict, OUT_OF_SCOPE_LABEL)
    css_class = verdict if verdict in VERDICT_LABELS else "REFUS"  # couleur de l'étiquette
    return f'<span class="v {css_class}">{label}</span>\n\n{result.get("feedback", "")}'


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
        # le cours s'affiche dans la page : le chat signale seulement la nouveauté
        titles = ", ".join(f"*{section['titre']}*" for section in sections)
        note += f"📖 La suite du cours vient d'apparaître : {titles}.\n\n"
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


def _gauge(label: str, percent: int, aria: str) -> str:
    """Une ligne de progression : libellé puis jauge horizontale."""
    return (
        f'<div class="bm-gauge-row"><span class="bm-gauge-label">{label}</span>'
        f'<div class="bm-progress-track" role="progressbar" aria-label="{aria}" aria-valuemin="0" '
        f'aria-valuemax="100" aria-valuenow="{percent}">'
        f'<span class="bm-progress-fill" style="width:{percent}%"></span></div></div>'
    )


def render_progress(state: dict) -> str:
    """Deux jauges empilées (questions, tentatives) pour l'en-tête de la console.

    Exemple :
        "Tentative 1 / 3" in render_progress(start_concept("monty-hall")[0])  # True
    """
    if not state.get("concept_id"):
        rows = [("Prêt à commencer", 0, "Progression du concept")]
    else:
        total = len(load_concept(state["concept_id"])["questions"])
        if state.get("finished"):
            rows = [(f"Concept terminé ({total} questions)", 100, "Progression du concept")]
        else:
            index = state["question_index"]
            attempt = min(state["attempts"] + 1, MAX_ATTEMPTS)
            rows = [
                (f"Question {index + 1} / {total}", round(100 * index / total), "Progression du concept"),
                # la jauge des tentatives montre l'essai en cours, déjà entamé
                (f"Tentative {attempt} / {MAX_ATTEMPTS}", round(100 * attempt / MAX_ATTEMPTS),
                 "Tentatives utilisées"),
            ]
    return '<div class="bm-progress-card">' + "".join(_gauge(*row) for row in rows) + "</div>"


def ordered_concepts() -> list[dict]:
    """Fiches disponibles, dans l'ordre de la maquette puis les nouvelles fiches.

    Exemple :
        ordered_concepts()[0]["id"]  # "monty-hall"
    """
    available = {concept_id for _, concept_id in list_concepts()}
    known = [concept_id for concept_id in CONCEPT_STYLES if concept_id in available]
    others = sorted(available - set(known))
    return [load_concept(concept_id) for concept_id in known + others]


def toggle_model(model_version: str) -> str:
    """Bascule entre les deux tuteurs.

    Exemple :
        toggle_model("baseline")  # "lora"
    """
    return "baseline" if model_version == "lora" else "lora"


# Repli de la console : purement visuel, géré dans le navigateur.
FOLD_JS = "() => { document.querySelector('#bm-console')?.classList.toggle('min'); }"
SCROLL_TOP_JS = "() => { window.scrollTo({top: 0}); }"


def build_app():
    """Construit la vue Gradio de la maquette : grille des histoires puis page de lecture.

    Le parcours et les appels au LLM restent ceux des fonctions ci-dessus.
    """
    import gradio as gr

    block_kwargs = {"title": "Brainmaxing"}
    if "theme" not in inspect.signature(gr.Blocks.launch).parameters:
        # Gradio 5 : thème, CSS et polices sont configurés sur Blocks.
        block_kwargs.update(theme=make_theme(gr), css=CUSTOM_CSS, head=HEAD_HTML)

    with gr.Blocks(**block_kwargs) as app:
        state = gr.State(_new_state())
        model_version = gr.State("baseline")

        # Page 1 : choix d'une histoire.
        with gr.Column(elem_id="bm-home") as home_view:
            gr.HTML(HEADER_HTML, container=False, padding=False)
            card_buttons = []
            with gr.Column(elem_id="bm-grid"):
                for concept in ordered_concepts():
                    with gr.Column(elem_id=f"bm-card-{concept['id']}", elem_classes=["bm-card"]):
                        gr.HTML(card_html(concept), container=False, padding=False)
                        hit = gr.Button(f"Commencer : {concept['id']}", elem_classes=["bm-hit"])
                    card_buttons.append((concept["id"], hit))

        # Page 2 : cours pleine largeur et console du tuteur en bas.
        with gr.Column(elem_id="bm-reading", visible=False) as reading_view:
            with gr.Row(elem_classes=["bm-topbar-row"]):
                back = gr.Button("Revenir aux histoires", elem_id="bm-back")
                gr.HTML(LOGO_HTML, container=False, padding=False)
            story = gr.HTML("", container=False, padding=False)
            course = gr.Markdown("", elem_classes=["bm-text"])
            with gr.Column(elem_id="bm-console"):
                with gr.Row(elem_classes=["bm-console-h"]):
                    fold = gr.Button("⤡", elem_id="bm-fold")
                    gr.HTML(console_title_html(), container=False, padding=False)
                    # Interrupteur Baseline / QLoRA : dessin HTML + bouton transparent par-dessus.
                    with gr.Column(elem_id="bm-tut"):
                        switch = gr.HTML(switch_html("baseline"), container=False, padding=False)
                        switch_hit = gr.Button("Changer de tuteur", elem_classes=["bm-hit"])
                    progress = gr.HTML(render_progress(_new_state()), container=False, padding=False)
                chatbot_kwargs = {"elem_id": "bm-chat", "show_label": False, "container": False}
                chatbot_params = inspect.signature(gr.Chatbot).parameters
                if "type" in chatbot_params:
                    chatbot_kwargs["type"] = "messages"
                if "buttons" in chatbot_params:
                    chatbot_kwargs["buttons"] = []
                chatbot = gr.Chatbot(**chatbot_kwargs)
                with gr.Row(elem_id="bm-composer"):
                    textbox = gr.Textbox(
                        placeholder="Explique avec tes mots…", show_label=False, container=False,
                        lines=1, max_lines=4, scale=1, elem_id="bm-answer",
                    )
                    send = gr.Button("↑", elem_id="bm-send", scale=0, min_width=34)

        def show(page):
            """Visibilité des deux pages : accueil ou lecture."""
            on_home = page == "home"
            return gr.update(visible=on_home), gr.update(visible=not on_home)

        pages = [home_view, reading_view]

        def on_open(concept_id):
            new_state, new_chat = start_concept(concept_id)
            concept = load_concept(concept_id)
            return (new_state, new_chat, story_header_html(concept), render_course(new_state),
                    render_progress(new_state), "", *show("read"))

        for concept_id, hit in card_buttons:
            hit.click(
                partial(on_open, concept_id), None,
                [state, chatbot, story, course, progress, textbox, *pages],
            ).then(None, js=SCROLL_TOP_JS)

        back.click(lambda: show("home"), None, pages).then(None, js=SCROLL_TOP_JS)

        def on_toggle(version):
            version = toggle_model(version)
            return version, switch_html(version)

        switch_hit.click(on_toggle, model_version, [model_version, switch])
        fold.click(None, js=FOLD_JS)

        def on_send(text, current_state, history, version):
            new_state, new_chat = submit_answer(text, current_state, history, version)
            return new_state, new_chat, "", render_progress(new_state), render_course(new_state)

        for trigger in (send.click, textbox.submit):
            trigger(on_send, [textbox, state, chatbot, model_version],
                    [state, chatbot, textbox, progress, course])
    return app


def launch_app(**launch_kwargs):
    """Lance Gradio avec la palette adaptée à la version installée."""
    import gradio as gr

    if "theme" in inspect.signature(gr.Blocks.launch).parameters:
        # Gradio 6 : thème et CSS sont configurés lors du lancement.
        launch_kwargs.setdefault("theme", make_theme(gr))
        launch_kwargs.setdefault("css", CUSTOM_CSS)
        launch_kwargs.setdefault("head", HEAD_HTML)
    return build_app().launch(**launch_kwargs)


if __name__ == "__main__":
    launch_app()
