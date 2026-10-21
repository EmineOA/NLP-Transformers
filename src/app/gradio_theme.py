"""Identité visuelle de l'interface Gradio Brainmaxing (maquette « Become a smart guy »).

Ce module ne contient que de la présentation : couleurs, polices, fragments HTML
et CSS. Le parcours pédagogique reste dans `gradio_app.py`.
"""
from html import escape

# Éclair jaune posé sur le cerveau du logo.
BOLT_SVG = (
    '<svg viewBox="0 0 20 28" aria-hidden="true"><path d="M13 1 3 15h6l-3 12 11-15h-6z" '
    'fill="#FFD43B" stroke="#B88700" stroke-width="1.4" stroke-linejoin="round"/></svg>'
)

# Habillage de chaque fiche : titre court, accroche, emoji et couleur de carte.
# L'ordre de ce dictionnaire est l'ordre d'affichage de la grille.
CONCEPT_STYLES = {
    "monty-hall": {"title": "Le problème de Monty Hall", "emoji": "🐐", "color": "--c-blue",
                   "hook": "Trois portes, une voiture. Changer de porte double tes chances."},
    "bateau-de-thesee": {"title": "Le bateau de Thésée", "emoji": "⛵", "color": "--c-violet",
                         "hook": "On remplace chaque planche. Est-ce encore le même bateau ?"},
    "plier-une-feuille-42-fois": {"title": "Plier une feuille 42 fois", "emoji": "🌕",
                                  "color": "--c-yellow", "dark_ink": True,
                                  "hook": "Une feuille de papier qui atteint la Lune."},
    "dilemme-du-tramway": {"title": "Le dilemme du tramway", "emoji": "🚋", "color": "--c-coral",
                           "hook": "Tirer le levier sauve cinq vies et en sacrifie une."},
    "infinis-plus-grands-que-l-infini": {"title": "Des infinis plus grands que l'infini",
                                         "emoji": "🌌", "color": "--c-indigo",
                                         "hook": "Certains infinis en contiennent d'autres."},
    "doute-de-descartes": {"title": "Le doute de Descartes", "emoji": "🎭", "color": "--c-green",
                           "hook": "Et si tout ce que tu vois était une illusion ?"},
}

SUBJECT_LABELS = {"mathematiques": "Maths", "philosophie": "Philo"}
MODEL_LABELS = {"baseline": "Baseline", "lora": "QLoRA"}


def concept_style(concept: dict) -> dict:
    """Habillage d'une fiche, avec un repli pour les fiches ajoutées plus tard.

    Paramètres :
        concept : fiche concept (contrat 1).

    Retour :
        {"title", "emoji", "color", "hook", "dark_ink"}.

    Exemple :
        concept_style({"id": "monty-hall", "concept": "..."})["emoji"]  # "🐐"
    """
    fallback = {"title": concept["concept"], "emoji": "💡", "color": "--c-blue", "hook": ""}
    return {"dark_ink": False, **fallback, **CONCEPT_STYLES.get(concept["id"], {})}


def difficulty_label(score: int) -> str:
    """Traduit la difficulté chiffrée (0-100) de la fiche en mot.

    Exemple :
        difficulty_label(45)  # "Moyen"
    """
    if score < 40:
        return "Facile"
    return "Moyen" if score < 70 else "Difficile"


def concept_tag(concept: dict) -> str:
    """Étiquette « Matière · Difficulté » affichée au-dessus des titres.

    Exemple :
        concept_tag({"matiere": "mathematiques", "difficulte": 45})  # "Maths · Moyen"
    """
    subject = SUBJECT_LABELS.get(concept["matiere"], concept["matiere"].capitalize())
    return f"{subject} · {difficulty_label(concept['difficulte'])}"


LOGO_HTML = (
    '<div class="bm-topbar"><div class="bm-logo">'
    f'<span class="bm-mark bm-emoji" aria-hidden="true">🧠{BOLT_SVG}</span>Brainmaxing</div></div>'
)

HEADER_HTML = (
    LOGO_HTML
    + '<div class="bm-cover"><p>Become the smartest guy in the room</p></div>'
    + f'<span class="bm-page-icon bm-emoji" aria-hidden="true">🧠{BOLT_SVG}</span>'
    + '<h1 class="bm-title">Choisis une histoire</h1>'
)

# Squelette de chargement : affiché dès l'ouverture, par-dessus la page, puis retiré
# en fondu quand les cartes, le CSS et les polices sont prêts. Évite l'aperçu brut de
# Gradio (textes sans style, polices de secours) pendant les premières secondes.
SKELETON_HTML = """
<style>
#bm-skeleton { position:fixed; inset:0; z-index:9999; background:#F5F8FA; overflow:hidden;
  padding:0 clamp(16px,3vw,40px); transition:opacity .35s ease; }
#bm-skeleton.bm-sk-out { opacity:0; pointer-events:none; }
#bm-skeleton .sk { background:#E6EDF1; border-radius:10px; position:relative; overflow:hidden; }
#bm-skeleton .sk::after { content:""; position:absolute; inset:0; transform:translateX(-100%);
  background:linear-gradient(90deg, transparent, rgb(255 255 255 / .75), transparent);
  animation:bm-shimmer 1.3s ease-in-out infinite; }
@keyframes bm-shimmer { to { transform:translateX(100%); } }
#bm-skeleton .sk-top { display:flex; align-items:center; gap:10px; height:54px; }
#bm-skeleton .sk-logo { width:28px; height:28px; border-radius:50%; }
#bm-skeleton .sk-name { width:120px; height:16px; }
#bm-skeleton .sk-cover { height:clamp(150px,20vw,210px); border-radius:0;
  margin:0 calc(-1 * clamp(16px,3vw,40px)); }
#bm-skeleton .sk-icon { width:84px; height:84px; border-radius:50%; margin-top:-46px; background:#DCE5EB; }
#bm-skeleton .sk-title { width:min(340px,70%); height:34px; margin:18px 0 28px; }
#bm-skeleton .sk-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:16px; }
#bm-skeleton .sk-card { height:180px; border-radius:22px; padding:20px; display:grid;
  grid-template-columns:52px 1fr; gap:14px; align-content:start; background:#EBF1F4; }
#bm-skeleton .sk-card .sk { background:#DCE5EB; }
#bm-skeleton .sk-emo { width:52px; height:52px; border-radius:16px; grid-row:span 4; }
#bm-skeleton .sk-l1 { width:40%; height:10px; } #bm-skeleton .sk-l2 { width:80%; height:18px; }
#bm-skeleton .sk-l3 { width:95%; height:12px; } #bm-skeleton .sk-btn { width:110px; height:34px; border-radius:99px; margin-top:6px; }
@media (prefers-reduced-motion: reduce) { #bm-skeleton .sk::after { animation:none; } }
</style>
<script>
(function () {
  var card = '<div class="sk-card"><div class="sk sk-emo"></div><div class="sk sk-l1"></div>'
    + '<div class="sk sk-l2"></div><div class="sk sk-l3"></div><div class="sk sk-btn"></div></div>';
  var el = document.createElement("div");
  el.id = "bm-skeleton";
  el.setAttribute("aria-hidden", "true");
  el.innerHTML = '<div class="sk-top"><div class="sk sk-logo"></div><div class="sk sk-name"></div></div>'
    + '<div class="sk sk-cover"></div><div class="sk sk-icon"></div><div class="sk sk-title"></div>'
    + '<div class="sk-grid">' + card.repeat(6) + "</div>";
  (document.body || document.documentElement).appendChild(el);
  var start = Date.now();
  function ready() {
    var cover = document.querySelector("#bm-home .bm-cover, #bm-reading .bm-cover");
    return cover && getComputedStyle(cover).backgroundColor !== "rgba(0, 0, 0, 0)";
  }
  function hide() {
    el.classList.add("bm-sk-out");
    setTimeout(function () { el.remove(); }, 400);
  }
  (function wait() {
    if (ready()) {
      var fonts = document.fonts ? document.fonts.ready : Promise.resolve();
      fonts.then(function () { setTimeout(hide, 120); });
    } else if (Date.now() - start < 15000) {
      requestAnimationFrame(wait);
    } else {
      hide();  // filet de sécurité : ne jamais bloquer l'interface
    }
  })();
})();
</script>
"""

HEAD_HTML = (
    SKELETON_HTML +
    """<script>
    (() => {
      // Un seul minuteur, même si Gradio recharge l'en-tête.
      clearTimeout(window.bmCardHopTimer);
      function scheduleHop() {
        window.bmCardHopTimer = setTimeout(hop, 4000 + Math.random() * 5000);
      }
      function hop() {
        // Ne pas déplacer une carte pendant que l'utilisateur la cible.
        const cards = [...document.querySelectorAll('#bm-home .bm-card')].filter(card => {
          const rect = card.getBoundingClientRect();
          return card.getClientRects().length && rect.bottom > 0 && rect.top < innerHeight
            && !card.matches(':hover, :focus-within');
        });
        if (!document.hidden && cards.length && !document.getElementById('bm-skeleton')) {
          const card = cards[Math.floor(Math.random() * cards.length)];
          // « translate » conserve le survol et le liseré animé existants.
          card.animate([
            {translate: '0 0'}, {translate: '0 -8px', offset: 0.4},
            {translate: '0 0', offset: 0.8}, {translate: '0 -2px', offset: 0.9},
            {translate: '0 0'}
          ], {id: 'bm-card-hop', duration: 650, easing: 'ease-in-out'});
        }
        scheduleHop();
      }
      scheduleHop();
    })();
    </script>"""
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:'
    'opsz,wght@12..96,700;12..96,800&family=Hanken+Grotesk:wght@400;500;600;700'
    '&family=JetBrains+Mono:wght@500;700&display=swap">'
    # Gradio retire @property du CSS personnalisé : sans cette déclaration, l'angle
    # animé du liseré des cartes est invalide et le fond des cartes disparaît.
    '<style>@property --a{syntax:"<angle>";inherits:false;initial-value:0deg}</style>'
)


def card_html(concept: dict) -> str:
    """Contenu d'une carte de la grille d'accueil (le clic est géré par Gradio).

    Exemple :
        "Commencer" in card_html(load_concept("monty-hall"))  # True
    """
    style = concept_style(concept)
    return (
        f'<div class="bm-card-in"><span class="bm-emo bm-emoji" aria-hidden="true">{style["emoji"]}</span>'
        f'<div><span class="bm-tag">{escape(concept_tag(concept))}</span>'
        f'<h3>{escape(style["title"])}</h3><p>{escape(style["hook"])}</p></div>'
        '<span class="bm-go">Commencer</span></div>'
    )


def story_header_html(concept: dict) -> str:
    """Bandeau coloré, icône et titre de la page de lecture d'une fiche."""
    style = concept_style(concept)
    ink = "#2A2208" if style["dark_ink"] else "#fff"
    return (
        f'<div class="bm-cover" style="background:var({style["color"]})">'
        f'<p style="color:{ink}">{escape(style["hook"] or style["title"])}</p></div>'
        f'<span class="bm-page-icon bm-emoji" aria-hidden="true">{style["emoji"]}</span>'
        f'<h1 class="bm-title bm-story-title">{escape(style["title"])}</h1>'
    )


def switch_html(model_version: str) -> str:
    """Interrupteur B/Q de la console : Baseline à gauche, QLoRA à droite."""
    label = MODEL_LABELS.get(model_version, "Baseline")
    return (
        f'<span class="bm-sw" data-v="{label}" role="switch" aria-checked="{str(label == "QLoRA").lower()}" '
        f'aria-label="Tuteur {label}"><span class="b">B</span><span class="q">Q</span></span>'
        f'<span class="bm-tip">Tuteur : {label} (cliquer pour changer)</span>'
    )


def console_title_html() -> str:
    """En-tête de la console : logo et nom du tuteur (la version est dans l'interrupteur)."""
    return (
        f'<div class="bm-console-title"><span class="bm-mark bm-emoji" aria-hidden="true">🧠{BOLT_SVG}</span>'
        '<b>Ton tuteur</b></div>'
    )


def _card_rules() -> str:
    """Couleur et décalage d'animation propres à chaque carte de la grille."""
    rules = []
    for index, (concept_id, style) in enumerate(CONCEPT_STYLES.items()):
        ink = "#2A2208" if style.get("dark_ink") else "#fff"
        rules.append(f"#bm-card-{concept_id}{{--c:var({style['color']});--ink:{ink};--i:{index}}}")
    return "\n".join(rules)


CUSTOM_CSS = """
:root, .gradio-container {
  --cover:#1E9ACD; --primary:#1C9ACE; --primary-ink:#0E6F99; --tint:#E8F5FB;
  --bg:#F5F8FA; --surface:#FFFFFF; --fill:#EEF3F6; --line:#D5E1E7; --fg:#0F1D24; --muted:#6A7F8A;
  --ok:#1E9C61; --warn:#B87900; --bad:#D0443A; --refus:#6C5BD0;
  --c-blue:#1C9ACE; --c-violet:#A15BE0; --c-yellow:#F4CD3C; --c-coral:#EC7A72; --c-green:#26A97A; --c-indigo:#4F67E8;
  --f-display:"Bricolage Grotesque", "Hanken Grotesk", system-ui, sans-serif;
  --f:"Hanken Grotesk", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-mono:"JetBrains Mono", ui-monospace, Consolas, monospace;
  --pad:clamp(16px,3vw,40px);
}
body, gradio-app { background:var(--bg) !important; }
.gradio-container {
  max-width:none !important; width:100% !important; margin:0 !important;
  padding:0 0 48px !important; background:var(--bg) !important;
  color:var(--fg); font:17px/1.55 var(--f) !important; color-scheme:light;
  -webkit-font-smoothing:antialiased; overflow-x:clip;
}
/* Gradio limite la page à 1280 px avec ses propres marges : la maquette est en pleine largeur. */
.gradio-container .main.fillable, .gradio-container .main.app { max-width:none !important; width:100% !important;
  margin:0 !important; padding:0 var(--pad) 0 var(--pad) !important; }
.gradio-container > .main, .gradio-container .wrap.contain { gap:0 !important; }
footer { display:none !important; }
.bm-emoji { font-family:"Apple Color Emoji","Segoe UI Emoji","Noto Color Emoji",sans-serif; line-height:1; }
.gradio-container :focus-visible { outline:3px solid var(--primary) !important; outline-offset:3px; }
.gradio-container .html-container, .gradio-container .prose { padding:0 !important; }

/* Logo, bandeau et titres */
.bm-topbar { display:flex; align-items:center; padding-block:14px; }
.bm-logo { display:flex; align-items:center; gap:10px; font:800 1.15rem var(--f-display);
  letter-spacing:-.02em; color:var(--fg); }
.bm-mark { position:relative; font-size:26px; }
.bm-mark svg { position:absolute; width:10px; height:14px; top:-1px; right:-1px;
  filter:drop-shadow(0 1px 1px rgb(0 0 0 / .25)); }
.bm-cover { margin-left:calc(-1 * var(--pad)); margin-right:calc(-1 * var(--pad)); height:clamp(150px,20vw,210px); background:var(--cover);
  display:flex; align-items:center; justify-content:center; padding-inline:16px; }
.bm-cover p { margin:0; max-width:22ch; color:#fff; font:800 clamp(1.8rem,1rem + 3.5vw,3.4rem)/1 var(--f-display);
  letter-spacing:-.03em; text-align:center; text-wrap:balance; }
.bm-page-icon { position:relative; display:inline-block; font-size:clamp(64px,8vw,84px);
  margin-top:calc(clamp(64px,8vw,84px) * -.55); }
.bm-page-icon svg { position:absolute; width:30%; height:42%; top:-2%; right:-2%;
  filter:drop-shadow(0 1px 2px rgb(0 0 0 / .3)); }
.bm-title { font:700 clamp(1.8rem,1.3rem + 2vw,2.4rem)/1.1 var(--f-display) !important; letter-spacing:-.02em;
  margin:8px 0 20px !important; color:var(--fg) !important; }
.bm-story-title { margin-top:18px !important; }  /* respiration sous l'emoji de l'histoire */

/* Flèche de retour à gauche du logo (page de lecture) */
.bm-topbar-row { align-items:center !important; gap:0 !important; flex-wrap:nowrap !important; }
.bm-topbar-row > * { flex:none !important; min-width:0 !important; width:auto !important; }
#bm-back { width:38px !important; height:38px; min-width:0 !important; padding:0 !important; margin-right:10px;
  border-radius:50% !important; border:0 !important; background:var(--surface) !important; color:var(--fg) !important;
  box-shadow:0 1px 4px rgb(15 29 36 / .12) !important; position:relative; overflow:hidden;
  color:transparent !important; font-size:0 !important; transition:background .2s,transform .15s; }
/* le libellé reste pour l'accessibilité ; la flèche est centrée par-dessus */
#bm-back::before { content:""; position:absolute; inset:0; margin:auto; width:20px; height:20px;
  background:var(--fg);
  -webkit-mask:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='none' stroke='black' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M16 10H4M9 4l-6 6 6 6'/%3E%3C/svg%3E") center/contain no-repeat;
  mask:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='none' stroke='black' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M16 10H4M9 4l-6 6 6 6'/%3E%3C/svg%3E") center/contain no-repeat; }
#bm-back:hover { background:var(--tint) !important; }
#bm-back:hover::before { background:var(--primary-ink); }
#bm-back:active { transform:scale(.94); }

/* Interrupteur Baseline / QLoRA dans l'en-tête de la console (même dessin, à l'horizontale) */
#bm-tut { position:relative; width:60px !important; height:32px; min-width:0 !important; border-radius:99px;
  display:grid !important; place-items:center; padding:0 !important; gap:0 !important; flex:none !important;
  overflow:visible !important; }
.bm-tip { display:none; }
#bm-tut:hover .bm-tip { display:block; position:absolute; top:calc(100% + 8px);  /* la console coupe ce qui dépasse en haut */ left:50%;
  transform:translateX(-50%); background:var(--fg); color:var(--bg); font:600 .8rem var(--f); padding:5px 9px;
  border-radius:8px; white-space:nowrap; pointer-events:none; z-index:5; }
.bm-sw { position:relative; width:60px; height:32px; border-radius:99px; background:var(--fill); display:flex;
  justify-content:space-around; align-items:center; font:700 .75rem var(--f-mono); color:var(--muted); }
.bm-sw::before { content:""; position:absolute; left:3px; top:3px; width:26px; height:26px; border-radius:50%;
  background:var(--primary-ink); box-shadow:0 1px 3px rgb(0 0 0 / .25);
  transition:transform .3s cubic-bezier(.3,1.4,.5,1); }
.bm-sw[data-v="QLoRA"]::before { transform:translateX(28px); }
.bm-sw span { position:relative; z-index:1; transition:color .2s; }
.bm-sw[data-v="Baseline"] .b, .bm-sw[data-v="QLoRA"] .q { color:#fff; }

/* Bouton transparent qui rend toute une carte (ou l'interrupteur) cliquable */
.bm-hit { position:absolute !important; inset:0; z-index:3; width:100% !important; height:100% !important;
  min-width:0 !important; opacity:0; cursor:pointer; border-radius:inherit !important; }

/* Page 1 : grille de cartes */
#bm-grid { display:grid !important; grid-template-columns:repeat(auto-fill,minmax(320px,1fr));
  grid-auto-rows:1fr; gap:16px !important; }  /* 1fr : toutes les lignes, donc toutes les cartes, ont la même hauteur */
/* La hauteur de la carte descend jusqu'au contenu, pour caler « Commencer » en bas. */
.bm-card { display:flex !important; flex-direction:column; }
.bm-card > .block:not(:has(.bm-hit)), .bm-card .html-container, .bm-card .prose { flex:1 1 auto; display:flex !important;
  flex-direction:column; height:100%; }
.bm-card-in { flex:1 1 auto; grid-template-rows:auto 1fr auto; }
.bm-card-in > .bm-emo { grid-row:1; }
.bm-card-in > div { grid-column:2; grid-row:1 / span 2; }
@media (max-width:420px) { #bm-grid { grid-template-columns:1fr; } }
.bm-card { --a:0deg; --c:var(--c-blue); --ink:#fff; position:relative; color:var(--ink); border-radius:22px !important;
  padding:20px !important; min-width:0 !important; border:1.5px solid transparent !important;
  background:linear-gradient(var(--c),var(--c)) padding-box,
    conic-gradient(from var(--a),transparent 0 60%,rgb(255 255 255 / .55) 80%,transparent 100%) border-box !important;
  box-shadow:0 10px 24px -14px var(--c); transition:transform .12s ease-out,box-shadow .2s;
  animation:bm-spin 8s linear infinite; animation-delay:calc(var(--i) * -.7s); }
.bm-card:hover { transform:translateY(-3px); box-shadow:0 16px 30px -14px var(--c); }
.bm-card:active { transform:scale(.98); }
@keyframes bm-spin { to { --a:360deg; } }
.bm-card-in { display:grid; grid-template-columns:52px minmax(0,1fr); gap:14px; align-items:start; color:var(--ink); }
.bm-emo { display:grid; place-items:center; width:52px; height:52px; border-radius:16px;
  background:rgb(255 255 255 / .22); font-size:30px; }
.bm-tag { font-size:.75rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; opacity:.85; }
.bm-card-in h3 { font:700 1.2rem/1.2 var(--f-display) !important; margin:2px 0 4px !important;
  letter-spacing:-.01em; text-wrap:balance; color:var(--ink) !important; }
.bm-card-in p { margin:0 !important; font-size:.92rem; opacity:.9; line-height:1.4; color:var(--ink) !important; }
.bm-go { grid-column:2; grid-row:3; align-self:end; margin-top:14px; justify-self:start; background:var(--surface); color:var(--fg); font-size:.88rem;
  font-weight:700; padding:8px 18px; border-radius:99px; box-shadow:0 2px 6px rgb(0 0 0 / .12); transition:transform .15s; }
.bm-card:hover .bm-go { transform:scale(1.04); }

/* Page 2 : lecture pleine largeur */
#bm-reading { padding-bottom:calc(33vh + 40px); }
#bm-reading > * { margin:0 !important; }
.bm-story-title { margin-bottom:0 !important; }
.bm-text { padding:4px 4px 0 !important; }  /* + l'écart natif de Gradio = 48px, comme la maquette */
.bm-text h2:first-child, .bm-text .prose > h2:first-child { margin-top:0 !important; }
.bm-text h2 { font:700 1.35rem/1.25 var(--f-display) !important; margin:26px 0 8px !important;
  letter-spacing:-.01em; color:var(--fg) !important; border:0 !important; }
.bm-text p, .bm-text li { max-width:75ch; font-size:1.06rem !important; color:var(--fg) !important; line-height:1.55; }
.bm-text p { margin:0 0 12px !important; }

/* Console du tuteur : carte flottante en bas */
#bm-console { position:fixed !important; left:50%; transform:translateX(-50%); bottom:16px; z-index:8;
  width:min(760px,calc(100% - 100px)) !important; max-height:33vh; min-height:200px; background:var(--surface);
  border-radius:24px; box-shadow:0 12px 40px -12px rgb(15 29 36 / .28),0 0 0 1px var(--line);
  display:flex !important; flex-direction:column; gap:0 !important; padding:0 !important; overflow:hidden; }
@media (max-width:600px) { #bm-console { width:calc(100% - 88px) !important; left:16px; transform:none; } }
#bm-console > * { width:auto !important; }
.bm-console-h { align-items:center !important; gap:8px !important; padding:12px 18px 4px; flex-wrap:nowrap !important;
  width:auto !important; }
.bm-console-h > * { min-width:0 !important; flex:none !important; width:auto !important; }
.bm-console-h > :last-child { margin-left:auto !important; }  /* progression calée à droite */
.bm-console-title { white-space:nowrap; }
#bm-fold { width:32px !important; height:32px; min-width:0 !important; padding:0 !important; margin-left:-6px;
  border-radius:50% !important; border:0 !important; background:var(--fill) !important; color:var(--muted) !important;
  box-shadow:none !important; font-size:15px !important; line-height:1; transition:background .2s,color .2s; }
#bm-fold:hover { background:var(--tint) !important; color:var(--primary-ink) !important; }
.bm-console-title { display:flex; align-items:center; gap:8px; font-size:.85rem; font-weight:600; color:var(--muted); }
.bm-console-title .bm-mark { font-size:20px; }
.bm-console-title b { color:var(--fg); font:700 .95rem var(--f-display); }
.bm-progress-card { display:flex; flex-direction:column; align-items:flex-end; gap:4px; font-size:.78rem;
  color:var(--muted); }
.bm-gauge-row { display:flex; align-items:center; gap:10px; white-space:nowrap; }
.bm-gauge-label { font-weight:600; }
.bm-gauge-row:first-child .bm-gauge-label { color:var(--fg); font-weight:700; }
.bm-progress-track { width:96px; height:6px; border-radius:99px; background:var(--fill); overflow:hidden; }
.bm-progress-fill { display:block; height:100%; border-radius:inherit; background:var(--primary); transition:width .3s ease; }
.bm-gauge-row + .bm-gauge-row .bm-progress-fill { background:var(--warn); }  /* tentatives : couleur distincte */
#bm-console.min { min-height:0; }
#bm-console.min #bm-chat, #bm-console.min #bm-composer { display:none !important; }
#bm-console.min .bm-console-h { padding-bottom:12px; }

/* Fil de discussion : bulles */
/* Le fil défile à l'intérieur de la console, comme dans la maquette (tiers bas de l'écran). */
#bm-chat { --chat-h:max(88px, calc(33vh - 116px)); flex:1 1 auto; min-height:0 !important; height:auto !important;
  max-height:var(--chat-h); overflow:hidden !important; border:0 !important;
  background:transparent !important; box-shadow:none !important; padding:0 !important; }
#bm-chat .bubble-wrap { max-height:var(--chat-h); overflow-y:auto !important; }
#bm-chat .bubble-wrap, #bm-chat .wrapper { background:transparent !important; }
#bm-chat .bubble-wrap { padding:8px 18px !important; }
#bm-chat .message-row { margin:0 0 8px !important; padding:0 !important; }
#bm-chat .message { padding:9px 14px !important; border-radius:18px !important; font-size:.98rem !important;
  border:0 !important; box-shadow:none !important; max-width:85%; }
#bm-chat .message * { font-size:.98rem; line-height:1.45; }
#bm-chat .message.bot, #bm-chat .bot-row .message { background:var(--fill) !important; color:var(--fg) !important;
  border-bottom-left-radius:6px !important; }
#bm-chat .message.user, #bm-chat .user-row .message { background:var(--primary-ink) !important; color:#fff !important;
  border-bottom-right-radius:6px !important; }
#bm-chat .message.user *, #bm-chat .user-row .message * { color:#fff !important; }
#bm-chat .message p { margin:0 0 6px !important; }
#bm-chat .message p:last-child { margin:0 !important; }
#bm-chat .avatar-container, #bm-chat .message-buttons-bot, #bm-chat .icon-button-wrapper { display:none !important; }
.v { display:block; font-size:.7rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
  margin-bottom:2px; color:var(--muted); }
.v.PARFAIT { color:var(--ok) !important; } .v.PARTIEL { color:var(--warn) !important; }
.v.INCORRECT { color:var(--bad) !important; } .v.REFUS { color:var(--refus) !important; }

/* Zone de saisie en pilule */
#bm-composer { width:auto !important; flex-wrap:nowrap !important; align-items:flex-end !important; gap:8px !important;
  margin:6px 12px 12px; background:var(--fill); border-radius:99px; padding:5px 5px 5px 16px; }
#bm-composer > * { min-width:0 !important; }
#bm-composer > #bm-answer, #bm-composer > .block:has(#bm-answer) { flex:1 1 auto !important; }
#bm-answer, #bm-answer .wrap, #bm-answer label { background:transparent !important; border:0 !important;
  box-shadow:none !important; padding:0 !important; }
#bm-answer textarea { background:transparent !important; border:0 !important; box-shadow:none !important;
  resize:none; padding:7px 0 !important; min-height:24px; font:17px/1.4 var(--f) !important; color:var(--fg) !important; }
#bm-answer textarea::placeholder { color:var(--muted) !important; opacity:1; }
#bm-send { width:34px !important; height:34px; min-width:0 !important; flex:none !important; padding:0 !important;
  border-radius:50% !important; border:0 !important; background:var(--primary-ink) !important; color:#fff !important;
  font-size:0 !important; box-shadow:none !important; transition:transform .1s;
  position:relative; }
/* Flèche pleine : le texte du bouton reste disponible pour l'accessibilité. */
#bm-send::before { content:""; position:absolute; inset:0; margin:auto; width:20px; height:20px;
  background:#fff; pointer-events:none;
  -webkit-mask:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M10 20V10H5a1 1 0 0 1-.7-1.7l7-7a1 1 0 0 1 1.4 0l7 7A1 1 0 0 1 19 10h-5v10a1 1 0 0 1-1 1h-2a1 1 0 0 1-1-1Z'/%3E%3C/svg%3E") center/contain no-repeat;
  mask:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M10 20V10H5a1 1 0 0 1-.7-1.7l7-7a1 1 0 0 1 1.4 0l7 7A1 1 0 0 1 19 10h-5v10a1 1 0 0 1-1 1h-2a1 1 0 0 1-1-1Z'/%3E%3C/svg%3E") center/contain no-repeat; }
#bm-send:active { transform:scale(.92); }

@media (max-width: 900px) {
  .bm-cover p { font-size:clamp(1.6rem,6vw,2.4rem); }
}
@media (prefers-reduced-motion: reduce) {
  .bm-card { animation:none; }
  .bm-card, .bm-go, .bm-sw::before, .bm-progress-fill { transition:none !important; }
}
@media (prefers-reduced-motion: no-preference) {
  #bm-chat .message-row, #bm-home, #bm-reading { animation:bm-in .35s cubic-bezier(.2,.8,.2,1); }
  @keyframes bm-in { from { opacity:0; transform:translateY(6px); } }
}
""" + _card_rules()


def make_theme(gr):
    """Thème clair de la maquette, identique en modes clair et sombre de Gradio.

    Gradio bascule en sombre selon le système : on fixe les mêmes valeurs dans
    les deux modes pour que la maquette garde ses contrastes.
    """
    values = {
        "body_background_fill": "#F5F8FA",
        "body_text_color": "#0F1D24",
        "body_text_color_subdued": "#6A7F8A",
        "background_fill_primary": "#FFFFFF",
        "background_fill_secondary": "#EEF3F6",
        "block_background_fill": "#FFFFFF",
        "block_border_color": "#D5E1E7",
        "block_label_background_fill": "#EEF3F6",
        "block_label_text_color": "#0F1D24",
        "block_title_text_color": "#0F1D24",
        "block_info_text_color": "#6A7F8A",
        "input_background_fill": "#EEF3F6",
        "input_border_color": "#D5E1E7",
        "input_border_color_focus": "#1C9ACE",
        "panel_background_fill": "#FFFFFF",
        "button_primary_background_fill": "#0E6F99",
        "button_primary_background_fill_hover": "#1C9ACE",
        "button_primary_text_color": "#FFFFFF",
        "button_secondary_background_fill": "#EEF3F6",
        "button_secondary_background_fill_hover": "#E8F5FB",
        "button_secondary_text_color": "#0F1D24",
        "color_accent_soft": "#E8F5FB",
    }
    both_modes = {**values, **{f"{key}_dark": value for key, value in values.items()}}
    return gr.themes.Base(
        primary_hue="sky",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Hanken Grotesk"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ).set(**both_modes)
