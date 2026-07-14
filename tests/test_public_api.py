import torch

import ternarylayers
from ternarylayers import ActModels, Bit, LinearModels, convert_to_ternary


def test_public_api_version() -> None:
    assert ternarylayers.__version__ == "0.1.0"


def test_linear_model_builds_and_runs() -> None:
    model = LinearModels.LinearAct(
        in_features=4,
        out_features=2,
        act=ActModels.ReLU(),
    ).build()

    output = model(torch.ones(3, 4))

    assert output.shape == (3, 2)


def test_convert_to_ternary_replaces_convertible_children() -> None:
    model = torch.nn.Sequential(Bit.Linear(4, 2), torch.nn.ReLU())

    converted = convert_to_ternary(model)

    assert converted is model
    assert isinstance(converted[0], Bit.LinearInfer)
    assert converted[0].weight.requires_grad is False

