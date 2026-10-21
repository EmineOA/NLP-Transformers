"""Identité visuelle de l'interface Gradio Brainmaxxing."""

HEADER_HTML = """
<section class="bm-hero" aria-label="Présentation Brainmaxxing">
  <div class="bm-hero-orb bm-orb-one" aria-hidden="true"></div>
  <div class="bm-hero-orb bm-orb-two" aria-hidden="true"></div>
  <div class="bm-hero-content">
    <span class="bm-eyebrow">LABORATOIRE D'APPRENTISSAGE</span>
    <h1>Brain<span>maxxing.</span></h1>
    <p>Explore un concept, défends ton raisonnement et progresse avec un tuteur IA
    qui t'aide à trouver, plutôt qu'à recopier, la réponse.</p>
    <div class="bm-tags" aria-label="Fonctionnalités">
      <span>6 concepts</span><span>17 questions</span>
      <span>Indices progressifs</span><span>Qwen &amp; QLoRA</span>
    </div>
  </div>
  <div class="bm-hero-mark" aria-hidden="true">✦</div>
</section>
"""

SIDEBAR_GUIDE_HTML = """
<div class="bm-guide">
  <div class="bm-guide-row"><span class="bm-guide-index">01</span>
    <div><strong>Explore</strong><p>Découvre les sections du concept au fil des questions.</p></div></div>
  <div class="bm-guide-row"><span class="bm-guide-index">02</span>
    <div><strong>Argumente</strong><p>Écris ton raisonnement, même si tu hésites.</p></div></div>
  <div class="bm-guide-row"><span class="bm-guide-index">03</span>
    <div><strong>Progresse</strong><p>Après un échec, utilise les indices pour réessayer.</p></div></div>
</div>
"""

FOOTER_HTML = """
<footer class="bm-footer">
  <span>Brainmaxxing · Mini-projet NLP &amp; Transformers</span>
  <span>Un verdict aide à progresser ; il peut rester imparfait.</span>
</footer>
"""

CUSTOM_CSS = """
/* Palette et disposition */
:root {
  --bm-surface: rgba(17, 27, 47, .89);
  --bm-border: rgba(180, 197, 233, .14);
  --bm-text: #eef3ff;
  --bm-muted: #a8b8d3;
  --bm-accent: #9387ff;
}
.gradio-container {
  max-width: 1460px !important;
  margin-inline: auto !important;
  padding: 24px 28px 32px !important;
  background:
    radial-gradient(ellipse at 8% 3%, rgba(101, 77, 226, .19), transparent 30%),
    radial-gradient(ellipse at 92% 34%, rgba(28, 181, 169, .11), transparent 27%),
    #0b1120 !important;
}
.bm-hero {
  position: relative;
  isolation: isolate;
  overflow: hidden;
  border: 1px solid rgba(169, 164, 255, .23);
  border-radius: 26px;
  padding: 32px 40px;
  margin-bottom: 22px;
  background: linear-gradient(115deg, rgba(70, 55, 141, .52), rgba(16, 36, 56, .96) 77%);
  box-shadow: 0 20px 60px rgba(0, 0, 0, .20);
}
.bm-hero-content { position: relative; z-index: 2; max-width: 790px; }
.bm-eyebrow { font-size: .73rem; letter-spacing: .19em; color: #bfc4ff; font-weight: 750; }
.bm-hero h1 { font-size: clamp(2.1rem, 4vw, 3.6rem); font-weight: 850;
  color: #f8faff; line-height: 1.12; margin: 12px 0 10px; letter-spacing: -.045em; }
.bm-hero h1 span { color: #a8a0ff; }
.bm-hero p { color: #d1ddf1; max-width: 660px; font-size: 1rem; line-height: 1.65; margin: 0; }
.bm-tags { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 21px; }
.bm-tags span { padding: 7px 12px; border-radius: 999px; font-weight: 600; font-size: .76rem;
  color: #dde7fa; background: rgba(255, 255, 255, .07);
  border: 1px solid rgba(255, 255, 255, .11); }
.bm-hero-orb { pointer-events: none; position: absolute; border-radius: 50%; filter: blur(3px); }
.bm-orb-one { width: 240px; height: 240px; right: -55px; top: -115px;
  background: radial-gradient(circle, rgba(145, 120, 255, .45), transparent 72%); }
.bm-orb-two { width: 250px; height: 250px; right: 5%; bottom: -178px;
  background: radial-gradient(circle, rgba(35, 207, 187, .32), transparent 72%); }
.bm-hero-mark { position: absolute; right: 7%; top: 40px; font-size: 110px; line-height: 1;
  color: rgba(225, 223, 255, .13); transform: rotate(-15deg); }

.bm-panel {
  background: var(--bm-surface) !important;
  border: 1px solid var(--bm-border) !important;
  border-radius: 22px !important;
  padding: 18px 18px 16px !important;
  box-shadow: 0 14px 42px rgba(0, 0, 0, .15);
  backdrop-filter: blur(12px);
}
.bm-side-title h3, .bm-main-title h3 { color: var(--bm-text) !important; margin-bottom: 3px; }
.bm-side-intro p, .bm-main-subtitle p { color: var(--bm-muted) !important; line-height: 1.55; }
.bm-field { margin-top: 7px; }
.bm-action-row { margin-top: 8px; }
.bm-panel .block { border-radius: 14px !important; }
.bm-panel button { border-radius: 13px !important; font-weight: 700 !important;
  transition: transform .16s ease, box-shadow .16s ease, background-color .16s ease !important; }
.bm-panel button:hover { transform: translateY(-2px); box-shadow: 0 7px 18px rgba(0, 0, 0, .17); }
.bm-primary button { box-shadow: 0 8px 25px rgba(99, 85, 244, .26) !important; }
.bm-guide { display: grid; gap: 17px; padding: 4px 1px 8px; }
.bm-guide-row { display: flex; align-items: flex-start; gap: 13px; }
.bm-guide-index { font: 750 .79rem/1.2 system-ui, sans-serif; color: #bfbaff;
  background: rgba(135, 121, 255, .14); border: 1px solid rgba(135, 121, 255, .25);
  border-radius: 10px; padding: 8px 9px; min-width: 37px; text-align: center; }
.bm-guide-row strong { font-size: .92rem; color: #f1f4ff; }
.bm-guide-row p { font-size: .81rem; line-height: 1.5; margin: 3px 0 0; color: var(--bm-muted); }
.bm-separator { height: 1px; background: var(--bm-border); margin: 16px 0; }
.bm-progress-card { padding: 13px 16px; border-radius: 15px;
  border: 1px solid rgba(157, 169, 212, .13);
  background: linear-gradient(105deg, rgba(74, 65, 150, .16), rgba(20, 44, 65, .22)); }
.bm-progress-head { display: flex; justify-content: space-between; gap: 8px; align-items: center; }
.bm-progress-title { color: #edf3ff; font-weight: 700; font-size: .88rem; }
.bm-progress-meta { color: #aebfda; font-size: .77rem; margin-top: 5px; }
.bm-progress-track { height: 7px; border-radius: 999px; overflow: hidden;
  background: rgba(191, 203, 232, .12); margin-top: 13px; }
.bm-progress-fill { display: block; height: 100%; border-radius: inherit;
  background: linear-gradient(90deg, #8072ff, #2bd3b5); transition: width .3s ease; }
.bm-chat { min-height: 500px; overflow: hidden; border-radius: 16px !important; }
.bm-input textarea { border-radius: 13px !important; }
.bm-footer { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px;
  margin: 20px 4px 0; color: #8f9db8; font-size: .77rem; }


/* Les blocs natifs de Gradio peuvent reprendre des surfaces du thème clair.
   Ces règles évitent les textes clairs sur des fonds blancs dans nos panneaux. */
.gradio-container {
  color-scheme: dark;
}
.gradio-container .bm-panel {
  color: var(--bm-text);
  color-scheme: dark;
}
.gradio-container .bm-panel .bm-side-title,
.gradio-container .bm-panel .bm-side-intro,
.gradio-container .bm-panel .bm-main-title,
.gradio-container .bm-panel .bm-main-subtitle {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
.gradio-container .bm-panel .bm-side-title *,
.gradio-container .bm-panel .bm-main-title * {
  color: #f1f5ff !important;
}
.gradio-container .bm-panel .bm-side-intro *,
.gradio-container .bm-panel .bm-main-subtitle * {
  color: #acbcda !important;
}
.gradio-container .bm-panel .bm-guide,
.gradio-container .bm-panel .bm-guide-row {
  background: transparent !important;
  color: #eaf0ff !important;
}
.gradio-container .bm-panel .bm-progress-card {
  background: linear-gradient(105deg, #26284f, #143042) !important;
  border: 1px solid #405078 !important;
  color: #edf3ff !important;
}
.gradio-container .bm-panel .bm-progress-title {
  color: #f4f7ff !important;
}
.gradio-container .bm-panel .bm-progress-meta {
  color: #bfcee7 !important;
}
.gradio-container .bm-panel input,
.gradio-container .bm-panel textarea,
.gradio-container .bm-panel select {
  color: #edf3ff !important;
  caret-color: #edf3ff;
  background-color: #0c182b !important;
  border-color: #324361 !important;
}
.gradio-container .bm-panel input::placeholder,
.gradio-container .bm-panel textarea::placeholder {
  color: #a8b8d3 !important;
  opacity: 1;
}
.gradio-container .bm-panel .bm-chat {
  background: #0c182b !important;
  color: #eaf0ff !important;
  border: 1px solid #30415e !important;
}
.gradio-container .bm-panel :is(.block-label, .block-title, label) {
  color: #e8eeff !important;
}
.gradio-container .bm-panel .info,
.gradio-container .bm-panel .wrap .info {
  color: #b2c0dc !important;
}
.gradio-container .bm-panel button:focus-visible,
.gradio-container .bm-panel input:focus-visible,
.gradio-container .bm-panel textarea:focus-visible {
  outline: 2px solid #b1a9ff !important;
  outline-offset: 2px;
}

@media (max-width: 900px) {
  .gradio-container { padding: 12px 13px 20px !important; }
  .bm-hero { padding: 24px 23px; border-radius: 19px; }
  .bm-hero-mark { display: none; }
  .bm-panel { padding: 13px !important; }
}
@media (prefers-reduced-motion: reduce) {
  .bm-panel button, .bm-progress-fill { transition: none !important; }
  .bm-panel button:hover { transform: none; }
}
"""


def make_theme(gr):
    """Applique des surfaces sombres cohérentes, en modes clair et sombre."""
    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="emerald",
        neutral_hue="slate",
        radius_size="lg",
    ).set(
        body_background_fill="#0b1120",
        body_background_fill_dark="#0b1120",
        body_text_color="#eef3ff",
        body_text_color_dark="#eef3ff",
        body_text_color_subdued="#a8b8d3",
        body_text_color_subdued_dark="#a8b8d3",
        background_fill_primary="#121d32",
        background_fill_primary_dark="#121d32",
        background_fill_secondary="#1a2740",
        background_fill_secondary_dark="#1a2740",
        block_background_fill="#111b2f",
        block_background_fill_dark="#111b2f",
        block_border_color="#293852",
        block_border_color_dark="#293852",
        block_label_background_fill="#202e4b",
        block_label_background_fill_dark="#202e4b",
        block_label_text_color="#e8eeff",
        block_label_text_color_dark="#e8eeff",
        block_title_background_fill="#111b2f",
        block_title_background_fill_dark="#111b2f",
        block_title_text_color="#eef3ff",
        block_title_text_color_dark="#eef3ff",
        block_info_text_color="#aebeda",
        block_info_text_color_dark="#aebeda",
        input_background_fill="#0c182b",
        input_background_fill_dark="#0c182b",
        input_background_fill_focus="#10223b",
        input_background_fill_focus_dark="#10223b",
        input_border_color="#324361",
        input_border_color_dark="#324361",
        input_border_color_focus="#9288ff",
        input_border_color_focus_dark="#9288ff",
        panel_background_fill="#111b2f",
        panel_background_fill_dark="#111b2f",
        panel_border_color="#293852",
        panel_border_color_dark="#293852",
        checkbox_label_background_fill="#1a2944",
        checkbox_label_background_fill_dark="#1a2944",
        checkbox_label_background_fill_selected="#5b50bd",
        checkbox_label_background_fill_selected_dark="#5b50bd",
        checkbox_label_text_color="#ebf0ff",
        checkbox_label_text_color_dark="#ebf0ff",
        checkbox_label_text_color_selected="#ffffff",
        checkbox_label_text_color_selected_dark="#ffffff",
        button_primary_background_fill="#6456ed",
        button_primary_background_fill_dark="#6456ed",
        button_primary_background_fill_hover="#776aff",
        button_primary_background_fill_hover_dark="#776aff",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#24344f",
        button_secondary_background_fill_dark="#24344f",
        button_secondary_background_fill_hover="#314366",
        button_secondary_background_fill_hover_dark="#314366",
        button_secondary_text_color="#eef3ff",
        button_secondary_text_color_dark="#eef3ff",
        accordion_text_color="#e5edff",
        accordion_text_color_dark="#e5edff",
    )
