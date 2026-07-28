import copy

import pytest
import torch

from toolkit.optimizer import (
    PRODIGY_PLUS_ALIASES,
    get_optimizer,
    optimizer_allows_external_gradient_clipping,
    optimizer_requires_eval_mode,
)


@pytest.mark.parametrize("alias", sorted(PRODIGY_PLUS_ALIASES))
def test_prodigy_plus_aliases_normalize_adam_style_group_lr(alias):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    groups = [{"params": [parameter], "lr": 1e-4}]

    optimizer = get_optimizer(
        groups,
        optimizer_type=alias,
        learning_rate=1e-4,
        optimizer_params={},
    )

    assert optimizer_requires_eval_mode(alias)
    assert not optimizer_allows_external_gradient_clipping(alias)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.01)


def test_prodigy_plus_keeps_explicit_relative_lr_and_overrides():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    groups = [{"params": [parameter], "lr": 0.5}]

    optimizer = get_optimizer(
        groups,
        optimizer_type="prodigy_plus",
        learning_rate=0.5,
        optimizer_params={"weight_decay": 0.2, "d_coef": 0.75},
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.2)
    assert optimizer.param_groups[0]["d_coef"] == pytest.approx(0.75)


def test_prodigy_plus_preserves_deliberately_frozen_group():
    trainable = torch.nn.Parameter(torch.tensor([1.0]))
    embedding = torch.nn.Parameter(torch.tensor([1.5]))
    frozen = torch.nn.Parameter(torch.tensor([2.0]))
    groups = [
        {"params": [trainable], "lr": 1e-4},
        {"params": [embedding], "lr": 5e-5},
        {"params": [frozen], "lr": 0.0},
    ]

    optimizer = get_optimizer(
        groups,
        optimizer_type="prodigy_plus",
        learning_rate=1e-4,
        optimizer_params={},
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1.0)
    assert optimizer.param_groups[2]["lr"] == pytest.approx(0.0)


def test_prodigy_plus_eval_train_round_trip_restores_raw_weights():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = get_optimizer(
        [parameter],
        optimizer_type="prodigy_plus",
        learning_rate=1.0,
        optimizer_params={},
    )
    parameter.grad = torch.tensor([0.25])
    optimizer.step()
    raw = parameter.detach().clone()

    optimizer.eval()
    optimizer.train()

    torch.testing.assert_close(parameter, raw, rtol=0, atol=0)


def test_prodigy_plus_resume_reconstructs_raw_weights_from_eval_save():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = get_optimizer(
        [parameter],
        optimizer_type="prodigy_plus",
        learning_rate=1.0,
        optimizer_params={},
    )
    for grad in (0.25, -0.1, 0.4):
        parameter.grad = torch.tensor([grad])
        optimizer.step()
        optimizer.zero_grad()

    raw = parameter.detach().clone()
    optimizer.eval()
    averaged = parameter.detach().clone()
    saved_state = copy.deepcopy(optimizer.state_dict())

    resumed_parameter = torch.nn.Parameter(averaged)
    resumed_optimizer = get_optimizer(
        [resumed_parameter],
        optimizer_type="prodigy_plus",
        learning_rate=1.0,
        optimizer_params={},
    )
    resumed_optimizer.load_state_dict(saved_state)
    resumed_optimizer.train()

    torch.testing.assert_close(resumed_parameter, raw, rtol=0, atol=0)


def test_standard_optimizer_does_not_require_eval_mode():
    assert not optimizer_requires_eval_mode("adamw8bit")
    assert not optimizer_requires_eval_mode(None)


def test_external_gradient_clipping_is_kept_for_adamw_but_not_adafactor():
    assert optimizer_allows_external_gradient_clipping("adamw")
    assert optimizer_allows_external_gradient_clipping("adamw8bit")
    assert optimizer_allows_external_gradient_clipping(None)
    assert not optimizer_allows_external_gradient_clipping("adafactor")
