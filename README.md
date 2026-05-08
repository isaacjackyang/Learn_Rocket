# Rocket Auto Research

Rocket Auto Research is a staged Auto Research framework for rocket GNC. The goal is not to hand-write one controller. The goal is to build a system that can generate strategies, run experiments, evaluate failures, keep research memory, and choose the next direction automatically.

## What this repo contains

- staged Auto Research loop
- shared GNC modules
- multiple strategy families
- surrogate research environment
- official `BalloonPoppingChallenge` environment adapter
- ActiveRocketPy cross-validation adapter
- HTML dashboard with runtime control and replay
- imitation learning and DAgger utilities

## What this repo does not contain

This repository is now kept intentionally slim.

It does **not** track these heavy third-party dependencies in git:

- `.external/BalloonPoppingChallenge/`
- `vendor/`
- `results/`

Those are treated as generated or external assets.

## Repository layout

```text
Rocket/
├─ configs/
│  ├─ auto_research.yaml
│  └─ legacy/
├─ experiments/
├─ rocket_auto_research/
├─ tests/
├─ .gitignore
├─ bootstrap_external.cmd
├─ clean_result.cmd
├─ pyproject.toml
├─ README.md
├─ start.cmd
└─ stop.cmd
```

## External dependencies

The official challenge and ActiveRocketPy are fetched on demand.

Bootstrap them with either:

```bat
bootstrap_external.cmd
```

or:

```bash
python experiments/bootstrap_external_dependencies.py
```

That script will:

1. clone `ARRC-Rocket/BalloonPoppingChallenge` into `.external/BalloonPoppingChallenge`
2. initialize its `ActiveRocketPy` submodule

After bootstrap, the repo expects these local paths to exist:

- `.external/BalloonPoppingChallenge`
- `.external/BalloonPoppingChallenge/ActiveRocketPy`

The code will also fall back to that `ActiveRocketPy` checkout for the high-fidelity adapter, so a separate `vendor/ActiveRocketPy` checkout is no longer required.

## Main architecture

### 1. Agent layer

- [rocket_auto_research/agents/competition_agent.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/agents/competition_agent.py)

Responsibilities:

- validate observation payloads
- parse them into `WorldState`
- call one selected strategy
- format and validate actions

Important boundary:

This is the repository's internal agent interface. It is **not** yet the official competition submission wrapper. The official challenge still expects an agent that inherits `BaseAgent` and implements `get_action(observation)`.

### 2. Shared GNC layer

- [rocket_auto_research/gnc](/F:/Documents/GitHub/Rocket/rocket_auto_research/gnc)

Key modules:

- `observation_parser.py`
- `state.py`
- `estimator.py`
- `target_selector.py`
- `guidance.py`
- `controller.py`
- `mission_manager.py`
- `safety_guard.py`

### 3. Strategy layer

- [rocket_auto_research/strategies](/F:/Documents/GitHub/Rocket/rocket_auto_research/strategies)

Current strategy families:

- `baseline_pid`
- `greedy_intercept`
- `predictive_intercept`
- `score_based`
- `mpc_light`
- `cem_planner`
- `energy_aware`
- `rl_policy_wrapper`
- `modular_strategy`

### 4. Auto Research layer

- [rocket_auto_research/auto_research](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research)

Core pieces:

- [problem_definition.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/problem_definition.py)
- [evaluator.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/evaluator.py)
- [failure_analyzer.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/failure_analyzer.py)
- [hypothesis_generator.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/hypothesis_generator.py)
- [research_memory.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/research_memory.py)
- [next_step_planner.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/next_step_planner.py)
- [mutation_engine.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/mutation_engine.py)
- [researcher_loop.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/researcher_loop.py)

The loop already supports:

- staged internal objectives
- promotion thresholds
- plateau detection
- stuck-mode diversification
- stage fallback
- kept-improvement tracking across experiments
- running-best evolution history with key-change annotations

## Staged optimization

The system internally decomposes the mission into stages:

- `launch_valid`
- `ascent_stable`
- `energy_margin`
- `approach_window`
- `balloon_pop`

Official competition reward is still a single mission score. Internally, the framework uses stage-specific dense objectives to make early research possible.

## Simulation backends

### Surrogate research backend

- [competition_platform_api.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/competition_platform_api.py)
- [competition_platform_adapter.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/competition_platform_adapter.py)

Used for:

- fast iteration
- stage bootstrapping
- safe CPU parallelism

### Official challenge backend

- [balloon_challenge_adapter.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/balloon_challenge_adapter.py)
- [balloon_challenge_loader.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/balloon_challenge_loader.py)

Used for:

- real `BalloonPoppingChallenge` Gym evaluation
- official observation/action contract research
- CPU parallel execution with per-worker isolated challenge runtime copies

If the official dependency checkout is missing, the adapter now raises a clear setup error telling you to run the bootstrap script first.

Implementation note:

The official challenge environment writes cache files into its package tree. To make multi-process workers safe, this repo now creates a per-process temporary runtime copy of `BalloonPoppingGymEnv` before import and execution.

### ActiveRocketPy backend

- [activerocketpy_adapter.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/activerocketpy_adapter.py)

Used for:

- higher-fidelity cross-validation
- 6-DoF behavior checks

If ActiveRocketPy is missing, the adapter now raises a clear setup error telling you to bootstrap external dependencies first.

## Main config

The main Auto Research entrypoint is:

- [configs/auto_research.yaml](/F:/Documents/GitHub/Rocket/configs/auto_research.yaml)

Older loop configs are kept only as historical references in:

- [configs/legacy](/F:/Documents/GitHub/Rocket/configs/legacy)

Single-strategy fixed configs remain in:

- [configs](/F:/Documents/GitHub/Rocket/configs)

## Dashboard

Main files:

- [rocket_auto_research/dashboard_builder.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/dashboard_builder.py)
- [rocket_auto_research/dashboard_server.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/dashboard_server.py)
- [experiments/run_dashboard_server.py](/F:/Documents/GitHub/Rocket/experiments/run_dashboard_server.py)

Features:

- bilingual dashboard
- day and night themes
- start, pause, resume, stop
- worker count control
- experiment-evolution chart with kept improvements and running-best staircase
- stage pipeline and promotion progress
- failure breakdown
- replay and timeline views

The experiment-evolution view is intended to answer one specific question:

- when experiment `N` improved the score, what changed?

Each completed experiment is shown in chronological order. The chart distinguishes:

- discarded experiments
- kept improvements
- running best over time

For each kept improvement, the dashboard annotates the most important change, for example:

- baseline
- strategy switch
- largest key parameter change

## Learning utilities

Scripts:

- [experiments/collect_imitation_dataset.py](/F:/Documents/GitHub/Rocket/experiments/collect_imitation_dataset.py)
- [experiments/train_behavioral_cloning.py](/F:/Documents/GitHub/Rocket/experiments/train_behavioral_cloning.py)
- [experiments/run_dagger_iteration.py](/F:/Documents/GitHub/Rocket/experiments/run_dagger_iteration.py)

## Typical workflow

### 1. Install the package

```bash
python -m pip install -e .
```

### 2. Fetch external dependencies

```bash
python experiments/bootstrap_external_dependencies.py
```

### 3. Run tests

```bash
python -m unittest discover -s tests -v
```

Tests that require official external repos will skip automatically if those dependencies are not installed.

### 4. Run Auto Research

```bash
python experiments/run_research_loop.py --config configs/auto_research.yaml
```

Worker count can also be set from the dashboard. New runs will apply the selected worker count to parallel-safe backends, including the official `balloon_challenge` adapter.

### 5. Start the dashboard

```bat
start.cmd
```

## Public maintenance policy

For `Learn_Rocket`, the intended policy is:

- keep original source code in repo
- keep generated outputs out of repo
- keep heavyweight third-party repos out of repo
- fetch official dependencies with bootstrap steps
- document setup clearly instead of vendoring large external trees

## Current limitations

- the repo still does not export a final official `BaseAgent` submission wrapper automatically
- official-env performance is still weaker than the research architecture itself
- stage definitions and hypothesis templates are still human-authored, not fully self-derived

## Official reference

Official challenge repository:

- [ARRC-Rocket/BalloonPoppingChallenge](https://github.com/ARRC-Rocket/BalloonPoppingChallenge)
