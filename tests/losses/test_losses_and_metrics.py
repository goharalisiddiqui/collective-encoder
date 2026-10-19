from types import SimpleNamespace
import pytest
import torch

from collective_encoder.losses.mse import CELossMSE
from collective_encoder.losses.nll import CELossNLL
from collective_encoder.losses.latent import CELossLatent
from collective_encoder.losses.mse_dict import CELossMSEDict
from collective_encoder.losses.steric import CELossSteric
from collective_encoder.losses.bond_deviation import CELossBondDeviation
from collective_encoder.losses.kld_uniform_gaussian import CELossKLDUniformGaussian
from collective_encoder.losses.kld_gaussian_mixture import CELossKLDGaussianMixture
from collective_encoder.losses.kld_schedulers.resolver import KLDResolver
from collective_encoder.metrics.mae import CEMetricMAE
from collective_encoder.metrics.mae_dict import CEMetricMAEDict
from collective_encoder.metrics.kld import CEMetricKLD


@pytest.mark.unit
def test_reconstruction_and_latent_losses(metargs):
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    x_hat = torch.tensor([[1.5, 2.5], [2.5, 3.5]], requires_grad=True)
    z = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], requires_grad=True)
    labels = torch.tensor([[0.0, 0.2], [0.3, 0.5]])

    # CELossMSE returns (loss, meta_dict)
    mse = CELossMSE(args={}, **metargs)
    loss_mse, meta_mse = mse(x, z[:2], x_hat, labels, {})
    assert torch.isclose(loss_mse, torch.tensor(0.25))
    assert meta_mse == {}
    loss_mse.backward(retain_graph=True)
    assert x_hat.grad is not None

    # CELossNLL returns (loss, meta_dict)
    nll = CELossNLL(args={}, **metargs)
    meta_nll = {"mu_x": x_hat, "logvar_x": torch.zeros_like(x_hat)}
    loss_nll, _ = nll(x, z[:2], x_hat, labels, meta_nll)
    assert torch.isfinite(loss_nll)

    # CELossLatent: sequential points [0,0], [1,0], [2,0] have step dist=1.0 and var=0.0
    lat_loss = CELossLatent(args={}, **metargs)
    loss_lat, _ = lat_loss(x, z, x_hat, labels, {})
    assert torch.isclose(loss_lat, torch.tensor(1.0), atol=1e-5)

    # CELossMSEDict compares output dict (3rd arg) against labels dict (4th arg)
    mse_dict = CELossMSEDict(
        args={"keys": ["a", "b"], "weights": [1.0, 2.0]},
        **metargs,
    )
    out_d = {"a": torch.tensor([2.0, 3.0]), "b": torch.tensor([5.0, 6.0])}
    lbl_d = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
    loss_d, sub_losses = mse_dict(None, z[:2], out_d, lbl_d, {})
    # MSE(a)=1.0 * 1.0 + MSE(b)=4.0 * 2.0 = 9.0
    assert torch.isclose(loss_d, torch.tensor(9.0))
    assert set(sub_losses.keys()) == {"a", "b"}


@pytest.mark.unit
def test_physics_losses_steric_and_bond_deviation(metargs):
    # 3 carbon atoms (Z=6, covalent radius ~ 0.76 A -> sum ~ 1.52 A) in 2 frames
    # Atom 0 and Atom 2 are at distance 0.2 A (< 0.5 * 1.52 A -> steric clash)
    # Bond (0, 1) has length 2.5 A (!= 1.52 A -> bond deviation)
    coords = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [0.0, 0.2, 0.0]],
            [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [0.0, 0.2, 0.0]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    z = torch.zeros((2, 2))
    bonds = [(0, 1), (0, 2)]

    steric = CELossSteric(
        args={"atomic_numbers": [6, 6, 6]},
        **metargs,
    )
    l_steric, _ = steric(coords, z, coords, None, {})
    assert l_steric > 0.0

    bond_dev = CELossBondDeviation(
        args={"atomic_numbers": [6, 6, 6], "bond_indices": bonds},
        **metargs,
    )
    l_bond, _ = bond_dev(coords, z, coords, None, {})
    assert l_bond > 0.0


@pytest.mark.unit
def test_kld_losses_and_schedulers(metargs):
    pl_stub = SimpleNamespace(
        current_epoch=5,
        trainer=SimpleNamespace(callback_metrics={"val_rec_loss": torch.tensor(0.5)}),
    )

    # Schedulers
    fixed_sched = KLDResolver("Fixed", {"value": 0.25}, **metargs)
    assert fixed_sched.get_kld_max() == 0.25

    lin_sched = KLDResolver(
        "Linear",
        {"start_value": 0.0, "end_value": 1.0, "start_epoch": 0, "end_epoch": 10},
        **metargs,
    )
    assert pytest.approx(lin_sched.get_kld_max()) == 0.1
    for _ in range(4):
        lin_sched.on_validation_epoch_end()
    assert pytest.approx(lin_sched.get_kld_max()) == 0.5

    auto_sched = KLDResolver(
        "Auto",
        {
            "kld_initial": 0.1,
            "kld_max": 1.0,
            "increase_factor": 0.5,
            "monitor_metric": "val_rec_loss",
        },
        **metargs,
    )
    assert pytest.approx(auto_sched.get_kld_max()) == 0.1
    # First epoch: metric improves from inf -> 0.5, value stays 0.1
    auto_sched.on_validation_epoch_end(pl_stub)
    assert pytest.approx(auto_sched.get_kld_max()) == 0.1
    # Second epoch with worse metric -> value increases by (1 + 0.5)
    pl_stub.trainer.callback_metrics["val_rec_loss"] = torch.tensor(0.8)
    auto_sched.on_validation_epoch_end(pl_stub)
    assert pytest.approx(auto_sched.get_kld_max()) == 0.15

    # Uniform Gaussian KLD: for mu=0, logvar=0, KLD is 0
    kld_ug = CELossKLDUniformGaussian(
        args={"kld_max_type": "Fixed", "kld_max_scheduler_args": {"value": 0.0}},
        plmodule=pl_stub,
        **metargs,
    )
    zero_meta = {
        "mu_latent": torch.zeros((4, 2)),
        "logvar_latent": torch.zeros((4, 2)),
    }
    val_zero, meta_zero = kld_ug(None, torch.zeros((4, 2)), None, None, zero_meta)
    assert torch.isclose(val_zero, torch.tensor(0.0), atol=1e-6)
    assert "kld" in meta_zero

    nonzero_meta = {
        "mu_latent": torch.ones((4, 2)),
        "logvar_latent": torch.zeros((4, 2)),
    }
    val_pos, _ = kld_ug(None, torch.ones((4, 2)), None, None, nonzero_meta)
    assert val_pos > 0.0

    # Gaussian Mixture KLD
    kld_gm = CELossKLDGaussianMixture(
        args={
            "n_components": 3,
            "kld_max_type": "Fixed",
            "kld_max_scheduler_args": {"value": 0.0},
        },
        plmodule=pl_stub,
        **metargs,
    )
    val_gm, _ = kld_gm(None, torch.zeros((4, 2)), None, None, nonzero_meta)
    assert torch.isfinite(val_gm)


@pytest.mark.unit
def test_kld_uniform_gaussian_linear_scheduler(metargs):
    pl_stub = SimpleNamespace(current_epoch=5)
    kld_ug = CELossKLDUniformGaussian(
        args={
            "kld_max_type": "Linear",
            "kld_max_scheduler_args": {
                "start_value": 0.0,
                "end_value": 1.0,
                "start_epoch": 0,
                "end_epoch": 10,
            },
        },
        plmodule=pl_stub,
        **metargs,
    )
    meta = {
        "mu_latent": torch.ones((4, 2)),
        "logvar_latent": torch.zeros((4, 2)),
    }
    val, out_meta = kld_ug(None, torch.ones((4, 2)), None, None, meta)
    assert torch.isfinite(val)
    assert "kld" in out_meta


@pytest.mark.unit
def test_metrics_mae_maedict_and_kld(metargs):
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_hat = torch.tensor([[2.0, 4.0], [4.0, 6.0]])

    mae = CEMetricMAE(args={}, **metargs)
    val_mae, meta_mae = mae(x, None, x_hat, None, {})
    assert torch.isclose(val_mae, torch.tensor(1.5))
    assert meta_mae == {}

    mae_dict = CEMetricMAEDict(args={"keys": ["k1", "k2"]}, **metargs)
    total_mae, res_dict = mae_dict(
        None,
        None,
        {"k1": x_hat, "k2": x},
        {"k1": x, "k2": x},
        {},
    )
    assert torch.isclose(total_mae, torch.tensor(1.5))
    assert torch.isclose(res_dict["k1"], torch.tensor(1.5))
    assert torch.isclose(res_dict["k2"], torch.tensor(0.0))

    kld_metric = CEMetricKLD(args={}, **metargs)
    meta = {
        "mu_latent": torch.zeros((3, 2)),
        "logvar_latent": torch.zeros((3, 2)),
    }
    val_kld, _ = kld_metric(None, None, None, None, meta)
    assert torch.isclose(val_kld, torch.tensor(0.0), atol=1e-6)
