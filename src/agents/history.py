class ConversationHistory:

    def __init__(self, max_turns: int = 3):
        self.max_turns = max_turns
        self.messages = []

    def add_user(self, content: str):
        self.messages.append({
            "role": "user",
            "content": content,
        })

        self._trim()

    def add_assistant(self, content: str):
        self.messages.append({
            "role": "assistant",
            "content": content,
        })

        self._trim()

    def get_messages(self):
        return list(self.messages)

    def clear(self):
        self.messages.clear()

    def _trim(self):
        max_messages = self.max_turns * 2

        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]