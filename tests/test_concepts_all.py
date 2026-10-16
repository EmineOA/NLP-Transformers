from pathlib import Path
from src.agents.concepts import load_concept
from src.llm.prompts import build_pedagogical_prompt

files = sorted(Path("data/concepts/json").glob("*.json"))
total_questions = 0

for path in files:
    fiche = load_concept(path.stem)

    for question in fiche["questions"]:
        prompt = build_pedagogical_prompt(
            fiche, question, version="fewshot"
        )
        assert question["enonce"] in prompt
        assert "[VERDICT: PARFAIT]" in prompt
        assert "[VERDICT: PARTIEL]" in prompt
        assert "[VERDICT: INCORRECT]" in prompt
        assert "[HORS_PERIMETRE]" in prompt

        total_questions += 1

    print(f"{fiche['id']} : {len(fiche['questions'])} questions OK")

assert len(files) == 6
assert total_questions == 17

print(f"\nValidation : {len(files)} fiches, {total_questions} questions")
