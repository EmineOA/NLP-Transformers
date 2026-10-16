"""Historique conversationnel borné (inchangé par rapport au dépôt)."""


class ConversationHistory:
    def __init__(self, max_turns: int = 3):
        if not isinstance(max_turns, int) or max_turns < 0:
            raise ValueError("max_turns doit être un entier positif ou nul")
        self.max_turns = max_turns
        self.messages = []

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def get_messages(self):
        return list(self.messages)

    def clear(self):
        self.messages.clear()

    def _trim(self):
        max_messages = self.max_turns * 2
        if max_messages == 0:
            self.messages.clear()
        elif len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]
