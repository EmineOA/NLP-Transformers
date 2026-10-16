
from src.agents.concepts import load_concept
from src.agents.baseline import BaselineAgent
from src.llm.model import load_model

# Charger le modèle et la fiche réelle
llm = load_model()
fiche = load_concept("monty-hall")
question = fiche["questions"][0]

# Créer l'agent pédagogique
agent = BaselineAgent(
    model=llm.model,
    tokenizer=llm.tokenizer,
    device=llm.device,
    fiche=fiche,
    question=question,
)

# Tester une réponse partiellement correcte
result = agent.ask(
    "Je garde parce qu'il reste deux portes, donc c'est 50/50."
)

print(result)
