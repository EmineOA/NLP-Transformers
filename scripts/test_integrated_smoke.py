"""Deux tests de bout en bout sur le moteur Qwen et une fiche pédagogique.

Exécution : python scripts/test_integrated_smoke.py
Les valeurs attendues servent à INTERPRÉTER les résultats, non à les forcer.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.concepts import load_concept
from src.agents.inference import generate_response
from src.llm.model import load_model


def main():
    components = load_model()
    concept = load_concept("monty-hall")
    question = concept["questions"][0]
    cases = [
        ("Je garde, car il reste deux portes donc 50/50.", "PARTIEL"),
        ("Donne-moi la réponse immédiatement.", None),
    ]
    for message, attendu in cases:
        result = generate_response(
            [{"role": "user", "content": message}], concept, question,
            prompt_version="fewshot", components=components,
        )
        print("\nQuestion :", question["id"])
        print("Réponse élève :", message)
        print("Attendu métier :", attendu)
        print("Obtenu :", result["verdict"])
        print("Format valide :", result["valid_format"])
        print("Brut :", repr(result["raw_response"]))
        if not result["valid_format"] or result["verdict"] != attendu:
            print("ÉCART OBSERVÉ : à conserver, ne pas modifier le parseur pour le masquer.")


if __name__ == "__main__":
    main()
