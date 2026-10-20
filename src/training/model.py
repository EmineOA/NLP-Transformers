"""Chargement du modèle de base en 4 bits, puis ajout des adaptateurs LoRA (bloc C).

Charge Qwen2.5-3B-Instruct en 4 bits (load_model_4bit), puis y ajoute des adaptateurs LoRA (add_lora).
Les poids sont "compressés" en 4 bits (quantization) : en 16 bits le modèle pèse environ 6 Go
de mémoire GPU ; en 4 bits, environ 2 Go.
Le modèle 4 bits reste gelé : seules les petites matrices A et B s'entraînent.
Une fois l'adaptateur sauvegardé, load_model_with_adapter le recharge sur un modèle 4 bits neuf.
"""

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.llm.model import DEFAULT_MODEL_NAME

# r : nombre de lignes de A et de colonnes de B.
# A réduit l'entrée à r valeurs, B les ramène à la taille de sortie.
LORA_RANK = 16
# alpha : force de la correction (elle est multipliée par alpha / r, ici 2).
LORA_ALPHA = 32
# Pendant l'entraînement, 5 % des valeurs en entrée de l'adaptateur sont mises à 0 au hasard,
# pour éviter qu'il apprenne par cœur.
LORA_DROPOUT = 0.05
# Toutes les couches linéaires des blocs Transformer, comme dans l'article QLoRA :
# 4 couches de l'attention (q, k, v, o) + 3 couches du MLP (gate, up, down).
# Pas lm_head, la couche finale qui choisit le token.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_model_4bit(model_name: str = DEFAULT_MODEL_NAME):
    """Charge le modèle de base en 4 bits sur le GPU, avec son tokenizer.

    Paramètres :
        model_name : nom du modèle sur Hugging Face (par défaut Qwen2.5-3B-Instruct).

    Retour :
        (model, tokenizer) ; le modèle est quantifié en 4 bits (NF4) et placé sur le GPU 0.

    Exemple :
        model, tokenizer = load_model_4bit()
        model.is_loaded_in_4bit  # True
    """
    # bitsandbytes (la bibliothèque qui fait le 4 bits) n'a pas de version CPU
    if not torch.cuda.is_available():
        raise RuntimeError("Aucun GPU CUDA détecté : le chargement en 4 bits (bitsandbytes) exige un GPU Nvidia.")

    # Les poids restent stockés en 4 bits ; ils sont décompressés en 16 bits juste le temps de chaque calcul.
    # bfloat16 sur la GPU pc ; float16 sur la T4 de Colab : la T4 (génération Turing)
    # n'a pas de support matériel du bfloat16, apparu avec les GPU Ampere.
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16

    # nf4 : un poids en 4 bits ne peut prendre que 16 valeurs ; nf4 les place là où se trouvent
    # la plupart des poids (près de 0) au lieu de les espacer régulièrement, donc on perd moins de précision.
    # Double quantization : les facteurs d'échelle de la compression sont eux-mêmes compressés
    # (≈0,4 bit économisé par poids, article QLoRA).
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # device_map={"": 0} : "" = tout le modèle (tous ses sous-modules), 0 = le GPU n°0
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map={"": 0},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def add_lora(model, rank: int = LORA_RANK, alpha: int = LORA_ALPHA, dropout: float = LORA_DROPOUT):
    """Ajoute les adaptateurs LoRA au modèle 4 bits gelé.

    LoRA : à côté de chaque grande matrice gelée W, on ajoute deux petites matrices A et B.
    La sortie devient : sortie = W·x + (alpha/r)·B·A·x. Seules A et B apprennent ; W ne bouge pas.
    r est la taille des petites matrices ; alpha est la force de la correction.

    Paramètres :
        model : modèle chargé en 4 bits par load_model_4bit.
        rank : valeur de r (taille des petites matrices).
        alpha : force de la correction.
        dropout : proportion de valeurs mises à 0 au hasard pendant l'entraînement (0.05 = 5 %).

    Retour :
        le modèle avec adaptateurs LoRA, prêt pour l'entraînement.

    Exemple :
        model = add_lora(model)
        any("lora_" in name for name, _ in model.named_parameters())  # True
    """
    # Prépare le modèle 4 bits à l'entraînement : passe en float32 les petites couches non compressées
    # (normalisations), pour la stabilité, et active le gradient checkpointing (on recalcule des valeurs
    # au lieu de les stocker : moins de mémoire GPU, un peu plus de temps).
    # Utile ici : nos conversations vont jusqu'à 3 638 tokens (MAX_LENGTH = 4096 dans dataset.py).
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",  # modèle qui prédit le token suivant
    )
    return get_peft_model(model, config)


def load_model_with_adapter(adapter_dir: str):
    """Charge le modèle de base en 4 bits, puis y branche un adaptateur LoRA sauvegardé.

    Paramètres :
        adapter_dir : dossier créé par model.save_pretrained après l'entraînement.

    Retour :
        (model, tokenizer) ; le modèle est en mode utilisation (eval), prêt à générer du texte.

    Exemple :
        model, tokenizer = load_model_with_adapter("adapters/test_5_exemples")
        any("lora_" in name for name, _ in model.named_parameters())  # True
    """
    # On ne sauvegarde que les petites matrices A et B (environ 115 Mo mesurés avec r = 16), pas le modèle de 6 Go.
    # Pour utiliser le tuteur, on recharge donc Qwen en 4 bits, puis on "branche" l'adaptateur dessus.
    # C'est ce que fera plus tard generate_response(model_version="lora").
    model, tokenizer = load_model_4bit()
    model = PeftModel.from_pretrained(model, adapter_dir)

    # mode utilisation : désactive le dropout, qui n'est utile que pendant l'entraînement ; les poids ne changent pas
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    # Lancer depuis la racine du dépôt : python -m src.training.model
    model, tokenizer = load_model_4bit()

    # is_loaded_in_4bit : état réel du modèle chargé ; quantization_config : réglage enregistré dans sa configuration
    assert model.is_loaded_in_4bit
    assert model.config.quantization_config.load_in_4bit
    assert next(model.parameters()).device.type == "cuda"

    memory_in_gb = torch.cuda.memory_allocated() / 1024**3  # octets -> Go
    applied_compute_dtype = model.config.quantization_config.bnb_4bit_compute_dtype
    print(f"modèle chargé en 4 bits sur {next(model.parameters()).device}")
    print(f"mémoire GPU utilisée : {round(memory_in_gb, 2)} Go")
    print(f"dtype de calcul : {applied_compute_dtype}")

    model = add_lora(model)

    # get_nb_trainable_parameters (peft) compte correctement les poids 4 bits, stockés à deux par octet
    trainable_count, total_count = model.get_nb_trainable_parameters()
    trainable_percent = 100 * trainable_count / total_count
    print(f"paramètres entraînables : {trainable_count} sur {total_count} ({trainable_percent:.2f} %)")
    print(f"mémoire GPU après add_lora : {round(torch.cuda.memory_allocated() / 1024**3, 2)} Go")

    assert 0 < trainable_percent < 5
    # requires_grad True = ce nombre sera modifié par l'entraînement ;
    # aucun poids du modèle de base ne doit l'être, seuls les adaptateurs apprennent
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert "lora_" in name, f"poids de base entraînable : {name}"

    # Test de rank : on repart d'un modèle neuf (add_lora ne se rappelle pas sur un modèle déjà adapté)
    # et on vérifie la forme réelle des matrices. A est (r, entrée) : sa 1re dimension vaut r ;
    # B est (sortie, r) : sa 2e dimension vaut r.
    del model
    torch.cuda.empty_cache()
    model, tokenizer = load_model_4bit()
    model = add_lora(model, rank=8)
    shape_a = next(param.shape for name, param in model.named_parameters() if "lora_A" in name)
    shape_b = next(param.shape for name, param in model.named_parameters() if "lora_B" in name)
    assert shape_a[0] == 8, f"lora_A devrait avoir 8 lignes, forme observée : {tuple(shape_a)}"
    assert shape_b[1] == 8, f"lora_B devrait avoir 8 colonnes, forme observée : {tuple(shape_b)}"
    print(f"rank=8 : lora_A {tuple(shape_a)}, lora_B {tuple(shape_b)} : OK")
