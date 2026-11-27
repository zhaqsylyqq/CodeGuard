"""Tiny tokenizer and causal LM for tests: no downloads, runs on CPU in seconds.

The tokenizer is byte-level (every character is a token, decoding is exact) and has a chat template
shaped like DeepSeek-Coder-instruct's: the assistant turn is closed by the EOS token plus a newline.
"""

TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}"
    "{% if m['role'] == 'user' %}### Instruction:\n{{ m['content'] }}\n"
    "{% else %}### Response:\n{{ m['content'] }}\n<eot>\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}### Response:\n{% endif %}"
)


def make_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<s>": 1, "<eot>": 2}
    for char in sorted(pre_tokenizers.ByteLevel.alphabet()):
        vocab[char] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>", eos_token="<eot>",
                                        pad_token="<pad>")
    tokenizer.chat_template = TEMPLATE
    return tokenizer


def make_model(tokenizer, seed=0):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                         max_position_embeddings=4096, bos_token_id=tokenizer.bos_token_id,
                         eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    return LlamaForCausalLM(config).eval()


def save_tiny_model(path):
    """Write a tiny model + tokenizer directory usable as --model in the scripts."""
    tokenizer = make_tokenizer()
    make_model(tokenizer).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path
