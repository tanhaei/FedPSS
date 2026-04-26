from fedpss.config import load_config
from fedpss.experiment import run_experiment


def test_one_federated_run(tmp_path):
    cfg = load_config("configs/test.yaml")
    cfg["experiment"]["n_patients"] = 48
    cfg["experiment"]["n_rounds"] = 1
    cfg["experiment"]["local_epochs"] = 1
    result = run_experiment(cfg, tmp_path)
    assert result["n_patients"] == 48
    assert "federated" in result["metrics"]
    assert (tmp_path / "metrics.json").exists()
