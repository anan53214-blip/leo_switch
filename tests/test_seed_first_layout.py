import json
import pytest

from scripts import run_multiuser_scaling_suite as suite


def test_seed_layout_and_method_identity(tmp_path):
    config = suite.config_from_args(suite.parse_args(["--seeds", "42", "43"]))
    for seed in config.seeds:
        paths = suite.build_paths(tmp_path, config.run_id, 20, seed=seed,
                                  results_root=config.results_root)
        assert paths.compare_output_dir == tmp_path / "results/multiseed_0907" / f"seed{seed}/u20"
        assert paths.system_run_dir == paths.compare_output_dir / "learned_baselines/han_pdqn"
        command = suite.build_train_command(paths, config, 20, seed)
        assert command[command.index("--algorithm") + 1] == "pdqn"
        assert "han_pdqn" not in config.baselines
        assert "pdqn" in config.baselines


def test_each_seed_gets_aggregate_and_root_gets_statistics(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(suite, "_generate_aggregate_artifacts",
                        lambda root, config, source, target: calls.append((config.seeds, source, target)) or [])
    config = suite.MultiUserConfig("test", seeds=(42, 43), results_root=str(tmp_path))
    suite.generate_aggregate_artifacts(tmp_path, config, tmp_path)
    assert calls == [((42,), tmp_path, tmp_path / "seed42"),
                     ((43,), tmp_path, tmp_path / "seed43"),
                     ((42, 43), tmp_path, tmp_path)]


def test_sca_merge_checks_protocol_and_only_adds_variant(tmp_path):
    sca = tmp_path / "sca"
    sca.mkdir()
    payload = {
        "environment_schema_version": suite.ENVIRONMENT_SCHEMA_VERSION,
        "metric_schema_version": 2,
        "env_config": {key: 0.1 for key in suite.REWARD_CONFIG_KEYS},
        "evaluation_seeds": [1000042],
        "system_checkpoint_sha256": "same-weight",
        "methods": [{"method": "han_pdqn", "reward": 1},
                    {"method": "han_pdqn_sca", "reward": 2}],
    }
    (tmp_path / "comparison_summary.json").write_text(json.dumps(payload))
    (sca / "comparison_summary.json").write_text(json.dumps(payload))
    (sca / "sca_manifest.json").write_text(json.dumps({"status": "complete", "checkpoint_sha256": "same-weight"}))
    rows = suite.read_sca_rows(tmp_path, 20, 42)
    assert [row["method"] for row in rows] == ["han_pdqn_sca"]
    payload["evaluation_seeds"] = [1000043]
    (sca / "comparison_summary.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="evaluation_seeds"):
        suite.read_sca_rows(tmp_path, 20, 42)


def test_missing_sca_is_optional(tmp_path):
    assert suite.read_sca_rows(tmp_path, 20, 42) == []


def test_seed_first_aggregation_reads_both_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "_plot_fixed_user_seed_summary", lambda *a, **kw: [])
    for seed, reward in [(42, 10), (43, 20)]:
        directory = tmp_path / f"seed{seed}/u20"
        history = directory / "learned_baselines/han_pdqn/training_history.json"
        history.parent.mkdir(parents=True)
        history.write_text("{}")
        (directory / "comparison_summary.csv").write_text("method,reward\nhan_pdqn,0\n")
        (directory / "comparison_summary.json").write_text(json.dumps({
            "environment_schema_version": suite.ENVIRONMENT_SCHEMA_VERSION,
            "metric_schema_version": 2,
            "env_config": {key: 0.1 for key in suite.REWARD_CONFIG_KEYS},
            "methods": [{"method": "han_pdqn", "reward": reward, "is_system": False}],
        }))
    rows, _ = suite.aggregate_user_summaries(tmp_path, [20], seeds=[42, 43],
                                             seed_first=True, proposed_method="han_pdqn")
    assert len(rows) == 1
    assert rows[0]["seed_count"] == "2"
    assert float(rows[0]["reward"]) == 15
    assert rows[0]["is_system"] == "True"


def test_history_directory_overrides_inherited_experiment_name(tmp_path):
    from scripts.plot_training_artifacts import _method_name, pretty_method_name
    path = tmp_path / "learned_baselines/pdqn/training_history.json"
    assert _method_name({"exp_name": "han_mappo_old", "algorithm": "pdqn"}, path) == "pdqn"
    assert pretty_method_name("han_pdqn_sca", False) == "SCA"
    assert pretty_method_name("unknown_experiment", True) == "unknown_experiment"
