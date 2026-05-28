from pathlib import Path

import torch

import scripts.generate_comparison_from_artifacts as artifacts
from scripts.generate_comparison_from_artifacts import sanitize_artifact_config_paths


def test_sanitize_artifact_config_paths_replaces_foreign_output_paths(tmp_path):
    config = {
        "save_path": "/home/pjpjq/LEO_switch/results/full_train_latency_priority_20260525_220324",
        "log_path": "/home/pjpjq/LEO_switch/results/logs",
    }

    sanitized = sanitize_artifact_config_paths(
        config,
        system_run_dir=Path("results/full_train_latency_priority_20260525_220324"),
        compare_dir=tmp_path / "baseline_compare",
    )

    assert sanitized["save_path"] == str(Path("results/full_train_latency_priority_20260525_220324"))
    assert sanitized["log_path"] == str(tmp_path / "baseline_compare" / "logs")
    assert config["save_path"].startswith("/home/")


def test_evaluate_maddpg_checkpoint_uses_current_algorithm_api(tmp_path, monkeypatch):
    checkpoint = tmp_path / "maddpg_model.pt"
    torch.save(
        {
            "config": {"device": "cpu"},
            "actor_state_dict": {"actor": torch.tensor([1.0])},
            "critic_state_dict": {"critic": torch.tensor([2.0])},
            "target_actor_state_dict": {"target_actor": torch.tensor([3.0])},
            "target_critic_state_dict": {"target_critic": torch.tensor([4.0])},
            "train_step": 7,
        },
        checkpoint,
    )

    class FakeModule:
        def __init__(self):
            self.loaded = None

        def load_state_dict(self, state):
            self.loaded = state

    class FakeAlgorithm:
        def __init__(self, config):
            self.config = config
            self.actor = FakeModule()
            self.critic = FakeModule()
            self.target_actor = FakeModule()
            self.target_critic = FakeModule()
            self.train_step = 0

    captured = {}

    def fake_evaluate_maddpg_policy(**kwargs):
        captured.update(kwargs)
        return {"method": "maddpg"}

    monkeypatch.setattr(artifacts, "MADDPGConfig", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(artifacts, "MADDPGAlgorithm", FakeAlgorithm, raising=False)
    monkeypatch.setattr(artifacts, "evaluate_maddpg_policy", fake_evaluate_maddpg_policy)

    result = artifacts.evaluate_maddpg_checkpoint(
        checkpoint=checkpoint,
        config_data={},
        objective="multi_objective",
        episodes=1,
        seed=42,
        max_steps=10,
        device_name="cpu",
    )

    assert "algorithm" in captured
    assert "actor" not in captured
    assert captured["algorithm"].actor.loaded == {"actor": torch.tensor([1.0])}
    assert captured["algorithm"].train_step == 7
    assert result["checkpoint"] == str(checkpoint)
