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

    assert filtered == ["han_attn", "attn_mappo"]


def test_pdqn_system_keeps_no_han_pdqn_baseline():
    assert compare.filter_duplicate_system_baselines(
        ["han_pdqn", "pdqn", "han_mappo"], {"algorithm": "pdqn"}
    ) == ["pdqn", "han_mappo"]


def test_pdqn_system_checkpoint_uses_offpolicy_evaluator(monkeypatch, tmp_path):
    captured = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"method": kwargs["method_name"], "is_system": False}

    def unexpected_mappo(**kwargs):
        raise AssertionError("PDQN must not load MAPPO actor/critic state dictionaries")

    monkeypatch.setattr(compare, "evaluate_han_offpolicy_checkpoint", fake_evaluate)
    monkeypatch.setattr(compare, "evaluate_mappo_checkpoint_with_trainer", unexpected_mappo)
    result = compare.evaluate_system_checkpoint(
        tmp_path / "best_model.pt", {"algorithm": "pdqn", "exp_name": "han_pdqn_u20"},
        "multi_objective", 1, "cpu", 2,
    )
    assert captured["trainer_cls"] is compare.HANPDQNTrainer
    assert result["is_system"] is True
    assert result["display_name"] == "HAN+PDQN"


def test_legacy_artifacts_without_algorithm_keep_mappo_routing():
    assert compare.system_trainer_class_for_config("multi_objective", {}) is compare.HANMAPPOTrainer
    assert compare.pretty_method_name("han_mappo_latency_priority_u20", True) == "HAN+MAPPO"


def test_pdqn_system_roundtrip_uses_its_own_encoder(tmp_path):
    """Synthetic weights check save/load and a real environment evaluation, not quality."""
    from dataclasses import asdict
    import torch
    from scripts.train import TrainConfig

    previous_threads = torch.get_num_threads()
    trainers = []
    torch.set_num_threads(1)
    try:
        config = TrainConfig(
            algorithm="mappo", exp_name="encoder_fixture", device="cpu",
            num_users=2, max_steps=2, replay_size=8, batch_size=2,
            save_path=str(tmp_path / "encoder"), log_path=str(tmp_path / "logs"),
        )
        encoder = compare.HANMAPPOTrainer(config)
        trainers.append(encoder)
        encoder._save_checkpoint(best=True)
        config = TrainConfig(
            algorithm="pdqn", exp_name="han_pdqn_fixture", device="cpu",
            num_users=2, max_steps=2, replay_size=8, batch_size=2,
            pretrained_han_path=str(tmp_path / "encoder/best_model.pt"),
            save_path=str(tmp_path / "system"), log_path=str(tmp_path / "logs"),
        )
        system = compare.HANPDQNTrainer(config)
        trainers.append(system)
        system._save_checkpoint(best=True)
        saved_config = asdict(config)
        saved_config["pretrained_han_path"] = str(tmp_path / "unavailable_encoder.pt")
        result = compare.evaluate_system_checkpoint(
            tmp_path / "system/best_model.pt", saved_config,
            "multi_objective", 1, "cpu", 2,
        )
        assert result["is_system"] is True
        assert result["display_name"] == "HAN+PDQN"
        assert result["episodes"] == 1
    finally:
        for trainer in trainers:
            trainer.env.close()
            for handler in trainer.logger.handlers[:]:
                handler.close()
                trainer.logger.removeHandler(handler)
        torch.set_num_threads(previous_threads)


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
