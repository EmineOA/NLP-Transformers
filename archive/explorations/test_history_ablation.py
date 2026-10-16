"""Teste l'effet de l'historique sur la question 2 du bateau de Thésée."""

from src.agents.concepts import load_concept
from src.agents.baseline import BaselineAgent
from src.llm.model import load_model

fiche = load_concept("bateau-de-thesee")
q2 = fiche["questions"][1]
q1_message = (
    "Non, ce n'est plus le même bateau. Pour moi, un objet reste le même "
    "seulement s'il conserve sa matière d'origine."
)
q2_message = (
    "Je choisis le bateau reconstruit avec les anciennes planches, "
    "selon ma règle précédente. Mais c'est gênant car ce bateau "
    "n'a pas conservé la continuité de son histoire."
)
llm = load_model()

def make_agent():
    """Crée un agent positionné directement sur la question 2."""
    return BaselineAgent(
        model=llm.model, tokenizer=llm.tokenizer, device=llm.device,
        fiche=fiche, question=q2,
    )

# A : même question sans information sur la règle précédente.
without_history = make_agent()
answer_a = without_history.ask(q2_message)

# B : même question avec la règle de Q1 et un retour corrigé.
# Ce retour est une donnée de contrôle, et non une sortie réelle de Qwen.
with_correct_history = make_agent()
with_correct_history.history.add_user(q1_message)
with_correct_history.history.add_assistant(
    "[VERDICT: PARFAIT]\nTa règle fondée sur la matière est cohérente."
)
answer_b = with_correct_history.ask(q2_message)

for label, result in (("A — sans historique", answer_a),
                      ("B — historique Q1 corrigé", answer_b)):
    print(f"\n{label}")
    print("Verdict :", result["verdict"])
    print("Format valide :", result["valid_format"])
    print("Feedback :", result["feedback"])
    print("Brut :", repr(result["raw_response"]))

assert with_correct_history.history.get_messages()[0]["content"] == q1_message
print("\nHistorique de contrôle transmis : OK")
