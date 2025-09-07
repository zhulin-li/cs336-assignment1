import torch
import numpy as np
import random


def load_data(
    data: np.array, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = []
    outputs = []
    for _ in range(batch_size):
        start = random.randint(a=0, b=len(data) - context_length - 1)
        end = start + context_length

        x = data[start:end]
        y = data[start + 1 : end + 1]
        inputs.append(torch.tensor(x))
        outputs.append(torch.tensor(y))

    inputs = torch.stack(inputs)
    outputs = torch.stack(outputs)

    inputs = inputs.to(device)
    outputs = outputs.to(device)

    return inputs, outputs
