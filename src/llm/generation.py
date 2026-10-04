import torch


@torch.no_grad()
def generate_chat(
    model,
    tokenizer,
    messages: list,
    device: torch.device,
    max_new_tokens: int = 200,
) -> str:

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    input_length = inputs["input_ids"].shape[1]

    new_tokens = generated[0, input_length:]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()