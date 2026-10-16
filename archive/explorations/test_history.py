
from src.agents.concepts import load_concept
from src.agents.baseline import BaselineAgent
from src.llm.model import load_model

llm = load_model()
fiche = load_concept("bateau-de-thesee")

agent = BaselineAgent(
    model=llm.model,
    tokenizer=llm.tokenizer,
    device=llm.device,
    fiche=fiche,
    question=fiche["questions"][0],
)

# Première question : l'étudiant définit sa règle.
response1 = agent.ask(
    "Non, ce n'est plus le même bateau. "
    "Pour moi, un objet reste le même seulement "
    "s'il conserve sa matière d'origine."
)

print("\nQUESTION 1")
print(response1)

# Passage à la deuxième question.
agent.set_question(fiche["questions"][1])

# L'étudiant fait référence à son raisonnement précédent.
response2 = agent.ask(
    "Je choisis le bateau reconstruit avec les "
    "anciennes planches, selon ma règle précédente. "
    "Mais c'est gênant car ce bateau n'a pas "
    "conservé la continuité de son histoire."
)

print("\nQUESTION 2")
print(response2)

print("\nHISTORIQUE")
for message in agent.history.get_messages():
    print(message["role"], ":", message["content"][:120])
