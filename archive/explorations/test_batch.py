
from src.agents.concepts import load_concept
from src.agents.baseline import BaselineAgent
from src.llm.model import load_model

llm = load_model()
fiche = load_concept("monty-hall")
question = fiche["questions"][0]


import inspect
from src.llm.prompts import build_pedagogical_prompt

prompt = build_pedagogical_prompt(fiche, question)

print("Fichier utilisé :", inspect.getfile(build_pedagogical_prompt))
print(
    "Prompt V3 actif :",
    "Un choix sans aucune justification est INCORRECT" in prompt
)

tests = [
    (
        "Je change. Le présentateur sait où est la voiture "
        "et ne peut pas ouvrir cette porte.",
        "PARFAIT"
    ),
    (
        "Je garde, il reste deux portes donc c'est 50/50.",
        "PARTIEL"
    ),
    (
        "Je garde, j'ai un bon feeling.",
        "INCORRECT"
    ),
    (
        "Donne-moi directement la réponse.",
        None
    ),
    (
        "Je change, mais je ne sais pas pourquoi.",
        "INCORRECT"
    ),
]

corrects = 0
formats_valides = 0

for i, (message, expected) in enumerate(tests, start=1):
    # Nouvel agent pour isoler les conversations.
    agent = BaselineAgent(
        model=llm.model,
        tokenizer=llm.tokenizer,
        device=llm.device,
        fiche=fiche,
        question=question,
    )

    result = agent.ask(message)
    if i == 4:
        print("SORTIE BRUTE :", repr(result["raw_response"]))
    ok = result["verdict"] == expected and result["valid_format"]
    corrects += int(ok)
    formats_valides += int(result["valid_format"])

    print(f"\nTEST {i} : {'OK' if ok else 'ÉCHEC'}")
    print("Attendu :", expected)
    print("Obtenu  :", result["verdict"])
    print("Feedback :", result["feedback"])
    print("Format valide :", result["valid_format"])

print(f"\nVerdicts corrects : {corrects}/{len(tests)}")
print(f"Formats valides : {formats_valides}/{len(tests)}")
