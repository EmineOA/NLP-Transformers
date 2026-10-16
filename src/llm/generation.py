"""Génération de réponses via le template conversationnel du tokenizer."""
import torch


@torch.no_grad()
def generate_chat(
    model, tokenizer, messages: list, device: torch.device,
    max_new_tokens: int = 160,
) -> str:
    """Génère en mode déterministe et retourne uniquement les tokens nouveaux."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False,
    ).to(device)
    generated = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    n_prompt = inputs["input_ids"].shape[1]
    return tokenizer.decode(generated[0, n_prompt:], skip_special_tokens=True).strip()
