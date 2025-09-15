from cs336_basics.blocks import TransformerLM, softmax
import torch


def decode_single_step(
    model: TransformerLM,
    inputs: torch.Tensor,
    temprature: float,
    nucleus_threshold: float | None,
) -> torch.Tensor:
    # inputs: (seq_len)
    # outputs: (seq_len, vocab_size)
    outputs = model(inputs)
    # logits: (vocab_size, )
    logits = outputs[:, -1, :]
    logits /= temprature
    # prob: (vocab_size, )
    probability = softmax(logits, dim=2)
    if nucleus_threshold is not None:

        def key(index: int):
            return -probability[index]

        sorted_index = sorted(range(len(probability)), key=key)
        for i, index in enumerate(sorted_index):
            if cum_prob < nucleus_threshold:
                cum_prob += probability[index]
            else:
                break
        probability[sorted_index[i:]] = 0
        probability /= probability.sum()
    next_token = torch.distributions.Categorical(probability).sample()
    return next_token


def decode(
    model: TransformerLM,
    inputs: torch.Tensor,
    temprature: float,
    nucleus_threshold: float | None,
    max_num_generated_tokens: int | None,
    end_token_id: int,
) -> torch.Tensor:
    num_generated_tokens = 0
    # inputs: (seq_len)
    while (
        max_num_generated_tokens is not None
        and num_generated_tokens < max_num_generated_tokens
    ):
        if inputs[-1] == end_token_id:
            return inputs
        num_generated_tokens += 1
        next_token = decode_single_step(model, inputs, temprature, nucleus_threshold)
        inputs = torch.concat(inputs, next_token)
    return inputs
