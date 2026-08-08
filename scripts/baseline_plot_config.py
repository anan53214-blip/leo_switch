"""Static metric and style configuration for baseline-comparison figures."""

from scripts.paper_metrics import (
    ADDITIONAL_METRICS,
    FIXED_CORE_METRICS,
)


CORE_BAR_METRICS = list(FIXED_CORE_METRICS)

TRAINING_QOS_STEP_METRICS = [
    ("mean_reward", "Reward", "Mean Episode Reward", 1.0),
    ("avg_success_delay", "Successful-Task Delay", "Average Delay (ms)", 1000.0),
    ("p95_success_delay", "P95 Successful-Task Delay", "P95 Delay (ms)", 1000.0),
    ("successful_task_throughput", "Successful Task Throughput", "Tasks / User-Minute", 1.0),
    ("task_success_rate", "Task Success", "Rate (%)", 100.0),
    ("deadline_violation_rate", "Deadline Violation", "Rate (%)", 100.0),
    ("handovers_per_user_minute", "Handover Frequency", "Handovers / User-Minute", 1.0),
    ("energy_per_successful_task", "Energy per Successful Task", "Energy / Task", 1.0),
    ("jain_mec_load_fairness", "MEC Load Jain Fairness", "Jain Index", 1.0),
]

REWARD_COMPONENT_STEP_METRICS = [
    ("mean_reward", "Total Reward", "Reward", 1.0),
    ("reward_task_success", "Task Success Reward", "Reward Term", 1.0),
    ("reward_load_balance", "MEC Load-Balance Reward", "Reward Term", 1.0),
    ("penalty_delay", "Delay Penalty", "Penalty Term", 1.0),
    ("penalty_energy", "Energy Penalty", "Penalty Term", 1.0),
    ("penalty_task_failure", "Task Failure Penalty", "Penalty Term", 1.0),
    ("penalty_service_interruption", "Service Interruption Penalty", "Penalty Term", 1.0),
    ("penalty_failed_handover", "Failed Handover Penalty", "Penalty Term", 1.0),
]

RADAR_METRICS = [
    ("successful_task_throughput", "Throughput", True),
    ("task_success_rate", "Task\nSuccess", True),
    ("avg_success_delay", "Low\nDelay", False),
    ("p95_success_delay", "Low P95\nDelay", False),
    ("deadline_violation_rate", "Deadline\nReliability", False),
    ("energy_per_successful_task", "Energy\nEfficiency", False),
    ("jain_mec_load_fairness", "MEC\nFairness", True),
]

ADDITIONAL_EPISODE_METRICS = list(ADDITIONAL_METRICS)

SYSTEM_STYLE = {
    "color": "#B03A2E",
    "linestyle": "-",
    "marker": "*",
    "linewidth": 3.0,
    "markersize": 11,
    "hatch": "///",
    "scatter_size": 280,
}

BASELINE_COLORS = [
    "#4E79A7",
    "#59A14F",
    "#9C755F",
    "#76B7B2",
    "#EDC948",
    "#BAB0AC",
    "#AF7AA1",
]
BASELINE_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
BASELINE_LINESTYLES = [
    "--",
    "-.",
    ":",
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (1, 1)),
    (0, (7, 2, 1, 2)),
]
BAR_HATCH_PATTERNS = ["///", "\\\\\\", "xx", "--", "oo", "++", "..", "**"]

LEARNED_BASELINE_COLORS = {
    "dqn": "#4E79A7",
    "maddpg": "#AF7AA1",
    "pdqn": "#EDC948",
    "han_mappo": "#E15759",
    "mappo_no_han": "#59A14F",
    "attn_mappo": "#1F77B4",
    "han_attn": "#B07AA1",
    "han_maddpg": "#17BECF",
    "han_pdqn": "#F28E2B",
}

SCATTER_LABEL_OFFSETS = {
    "HAN+MAPPO": (12, -16),
    "Random": (10, 8),
    "Min-Distance": (10, 12),
    "Full-Local": (10, -10),
    "Joint Greedy": (10, -12),
    "DQN": (10, 10),
    "MADDPG": (10, 12),
    "PDQN": (10, 12),
    "MAPPO": (10, -14),
    "HAN+MADDPG": (10, 12),
    "HAN+PDQN": (10, 12),
}

PAPER_COLORS = {
    "primary": "#0F4C81",
    "secondary": "#B03A2E",
    "success": "#1E8449",
    "warning": "#AF601A",
    "info": "#2471A3",
    "dark": "#283747",
    "muted": "#7B7D7D",
    "fill_alpha": 0.16,
}
