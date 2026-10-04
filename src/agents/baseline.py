import re

from src.agents.history import ConversationHistory
from src.llm.generation import generate_chat
from src.llm.prompts import build_system_prompt


VERDICT_PATTERN = re.compile(
    r"\[VERDICT:\s*([^\]]+)\]",
    re.IGNORECASE,
)


ALLOWED_VERDICTS = {
    "ADMISSIBLE",
    "NON_ADMISSIBLE",
    "INDETERMINE",
}


def extract_verdict(response: str) -> str | None:
    match = VERDICT_PATTERN.search(response)

    if match is None:
        return None

    verdict = match.group(1).strip().upper()

    if verdict not in ALLOWED_VERDICTS:
        return None

    return verdict


class BaselineAgent:

    def __init__(
        self,
        model,
        tokenizer,
        device,
        rules: str,
        fiche: str,
        max_history_turns: int = 3,
        system_prompt: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.system_prompt = (
            system_prompt
            if system_prompt is not None
            else build_system_prompt(
                rules=rules,
                fiche=fiche,
            )
        )

        self.history = ConversationHistory(
            max_turns=max_history_turns
        )

    def ask(self, user_message: str):

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

        messages.extend(
            self.history.get_messages()
        )

        messages.append({
            "role": "user",
            "content": user_message,
        })

        response = generate_chat(
            model=self.model,
            tokenizer=self.tokenizer,
            messages=messages,
            device=self.device,
        )

        verdict = extract_verdict(response)

        self.history.add_user(user_message)
        self.history.add_assistant(response)

        return {
            "response": response,
            "verdict": verdict,
        }