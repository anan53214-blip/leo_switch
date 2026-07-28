from pathlib import Path

import scripts.compare_system_baselines as compare
from scripts.compare_system_baselines import PROJECT_ROOT, train_config_from_dict


def test_train_config_from_dict_rewrites_stale_log_path_with_save_path(tmp_path):
    config = train_config_from_dict(
        {
            "save_path": "/home/pjpjq/LEO_switch/results/full_train_latency_priority",
            "log_path": "/home/pjpjq/LEO_switch/results/logs",
        },
        device="cpu",
        max_steps=600,
        episodes=3,
        save_path=tmp_path / "local_run",
    )

    assert Path(config.save_path) == tmp_path / "local_run"
    assert Path(config.log_path) == PROJECT_ROOT / "results" / "logs"


def test_filter_duplicate_system_baselines_removes_current_system():
    filtered = compare.filter_duplicate_system_baselines(
        ["han_attn", "attn_mappo"],
        {"algorithm": "han_attn"},
    )

    assert filtered == ["attn_mappo"]


def test_filter_duplicate_system_baselines_keeps_other_systems_intact():
    filtered = compare.filter_duplicate_system_baselines(
        ["han_attn", "attn_mappo", "han_mappo"],
        {"algorithm": "mappo"},
    )

    assert filtered == ["han_attn", "attn_mappo", "han_mappo"]


def test_no_han_mappo_can_reuse_existing_checkpoint_without_training(tmp_path, monkeypatch):
    save_dir = tmp_path / "learned_baselines" / "mappo_no_han"
    save_dir.mkdir(parents=True)
    checkpoint = save_dir / "best_model.pt"
    checkpoint.write_bytes(b"checkpoint")

    class UnexpectedTrainer:
        pass

    captured = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"method": "mappo_no_han"}

    monkeypatch.setattr(compare, "no_han_trainer_class_for_objective", lambda objective: UnexpectedTrainer)
    monkeypatch.setattr(compare, "evaluate_mappo_checkpoint_with_trainer", fake_evaluate)

    result = compare.train_and_evaluate_no_han_mappo(
        config_data={},
        objective="multi_objective",
        output_dir=tmp_path,
        device="cpu",
        episodes=3,
        max_steps=600,
        total_timesteps=300000,
        early_stop_patience=0,
        reuse_checkpoint_if_available=True,
    )

    assert captured["checkpoint"] == checkpoint
    assert captured["trainer_cls"] is UnexpectedTrainer
    assert result["source"] == "mappo_no_han_checkpoint_eval"
    assert captured["config_data"]["algorithm"] == "mappo"


def test_han_mappo_baseline_can_reuse_existing_checkpoint_without_training(tmp_path, monkeypatch):
    save_dir = tmp_path / "learned_baselines" / "han_mappo"
    save_dir.mkdir(parents=True)
    checkpoint = save_dir / "best_model.pt"
    checkpoint.write_bytes(b"checkpoint")

    class UnexpectedTrainer:
        pass

    captured = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"method": "han_mappo"}

    monkeypatch.setattr(compare, "trainer_class_for_objective", lambda objective: UnexpectedTrainer)
    monkeypatch.setattr(compare, "evaluate_mappo_checkpoint_with_trainer", fake_evaluate)

    result = compare.train_and_evaluate_han_mappo(
        config_data={"algorithm": "han_attn"},
        objective="multi_objective",
        output_dir=tmp_path,
        device="cpu",
        episodes=3,
        max_steps=600,
        total_timesteps=300000,
        early_stop_patience=0,
        reuse_checkpoint_if_available=True,
    )

    assert captured["checkpoint"] == checkpoint
    assert captured["trainer_cls"] is UnexpectedTrainer
    assert result["source"] == "han_mappo_checkpoint_eval"
    assert captured["config_data"]["algorithm"] == "mappo"
