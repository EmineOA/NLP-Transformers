import torch


@torch.no_grad()
def greedy_generate(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 30,
) -> str:
    """
    Génération autoregressive simple utilisant greedy decoding.

    À chaque étape :
    1. le modèle calcule les logits ;
    2. on récupère les logits du dernier token ;
    3. on choisit le token avec argmax ;
    4. on ajoute le token à la séquence ;
    5. on recommence.
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    for _ in range(max_new_tokens):

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        last_logits = outputs.logits[:, -1, :]

        next_token_id = torch.argmax(
            last_logits,
            dim=-1,
            keepdim=True,
        )

        input_ids = torch.cat(
            [input_ids, next_token_id],
            dim=-1,
        )

        next_attention = torch.ones(
            (attention_mask.shape[0], 1),
            dtype=attention_mask.dtype,
            device=device,
        )

        attention_mask = torch.cat(
            [attention_mask, next_attention],
            dim=-1,
        )

        if (
            tokenizer.eos_token_id is not None
            and next_token_id.item()
            == tokenizer.eos_token_id
        ):
            break

    return tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True,
    )

def apply_temperature(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature doit être > 0")

    return logits / temperature


def apply_top_k(
    logits: torch.Tensor,
    k: int,
) -> torch.Tensor:
    if k <= 0:
        return logits

    k = min(k, logits.shape[-1])

    threshold = torch.topk(logits, k).values[..., -1, None]

    return logits.masked_fill(
        logits < threshold,
        float("-inf"),
    )


def apply_top_p(
    logits: torch.Tensor,
    p: float,
) -> torch.Tensor:
    if not 0 < p <= 1:
        raise ValueError("top_p doit être dans ]0, 1]")

    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
    )

    sorted_probs = torch.softmax(
        sorted_logits,
        dim=-1,
    )

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=-1,
    )

    # On garde également le premier token
    # qui fait dépasser le seuil p.
    remove_mask = cumulative_probs > p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(
        remove_mask,
        float("-inf"),
    )

    filtered_logits = torch.full_like(
        logits,
        float("-inf"),
    )

    filtered_logits.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_logits,
    )

    return filtered_logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:

    logits = apply_temperature(
        logits,
        temperature,
    )

    logits = apply_top_k(
        logits,
        top_k,
    )

    logits = apply_top_p(
        logits,
        top_p,
    )

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    return torch.multinomial(
        probabilities,
        num_samples=1,
    )

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> str:

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)

    for _ in range(max_new_tokens):
        outputs = model(input_ids=input_ids)

        logits = outputs.logits[:, -1, :]

        next_token = sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=-1,
        )

        if (
            tokenizer.eos_token_id is not None
            and next_token.item() == tokenizer.eos_token_id
        ):
            break

    return tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True,
    )