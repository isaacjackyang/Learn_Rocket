# Rocket Auto Research

Rocket Auto Research is a staged Auto Research system for rocket GNC. The goal is not to hand-write one controller, but to build a closed-loop research framework that can generate strategies, run simulations, evaluate outcomes, remember failures, and plan the next experiments.

This repository currently supports:

- staged Auto Research with internal phase objectives
- multiple strategy families for target selection, guidance, and control
- surrogate and official-like simulation backends
- direct integration with the official `BalloonPoppingChallenge` Gym environment
- HTML dashboard with replay and runtime control
- imitation learning and DAgger utilities

## Core idea

The project is organized around two layers:

1. `competition agent`
   - executes one chosen strategy on one observation at a time
2. `auto research system`
   - proposes, tests, scores, mutates, and replaces strategies over time

The system is intended to answer questions like:

- which strategy family survives official scenarios more reliably
- which parameters improve ascent, energy, or intercept geometry
- which failure modes dominate the current stage
- whether the search is progressing or plateauing
- when the system should advance, fall back, or diversify

## Current status

The repository has a working staged Auto Research loop. It can:

- route different research stages to different adapters
- evaluate candidates with both mission fitness and stage fitness
- track stage success and promotion criteria
- detect plateaus
- enter stuck-mode diversification
- fall back to an earlier stage when a later stage stalls with heavy crashes

What it is not yet:

- a fully autonomous research scientist that invents new methods by itself
- a submission-ready official competition agent out of the box

The staged research framework is real. The final official submission wrapper is still a separate task.

## Repository layout

```text
Rocket/
├─ configs/
│  ├─ auto_research.yaml
│  ├─ aggressive.yaml
│  ├─ baseline.yaml
│  ├─ energy_aware.yaml
│  ├─ energy_transfer_robust.yaml
│  ├─ rl_policy_wrapper_bc.yaml
│  ├─ rl_policy_wrapper_mlp.yaml
│  ├─ rl_policy_wrapper_mlp_dagger.yaml
│  ├─ robust.yaml
│  └─ legacy/
├─ experiments/
├─ results/
├─ rocket_auto_research/
│  ├─ agents/
│  ├─ auto_research/
│  ├─ gnc/
│  ├─ learning/
│  ├─ strategies/
│  ├─ dashboard_builder.py
│  ├─ dashboard_server.py
│  └─ replay_enrichment.py
├─ tests/
├─ .external/
├─ vendor/
├─ start.cmd
├─ stop.cmd
└─ clean_result.cmd
```

## Architecture

### 1. Agent layer

Main file:

- [rocket_auto_research/agents/competition_agent.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/agents/competition_agent.py)

Responsibilities:

- validate incoming observation payloads
- parse them into internal `WorldState`
- call one selected strategy
- format and validate outgoing actions

Important note:

This is the repository's internal agent interface. It is not yet the official `BalloonPoppingChallenge` submission wrapper, because the official repo expects an agent that inherits `BaseAgent` and implements `get_action(observation)`.

## 2. Shared GNC layer

Main directory:

- [rocket_auto_research/gnc](/F:/Documents/GitHub/Rocket/rocket_auto_research/gnc)

Key modules:

- [observation_parser.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/gnc/observation_parser.py)
  - converts observation payloads into internal state objects
- [state.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/gnc/state.py)
  - defines `RocketState`, `BalloonState`, and `WorldState`
- [estimator.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/gnc/estimator.py)
  - supports `simple`, `alpha_beta`, `wind_aware`, and `jitter_aware`
- [target_selector.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/gnc/target_selector.py)
  - supports `nearest`, `score_based`, `reachable`, and energy-aware logic
- [guidance.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/gnc/guidance.py)
  - supports `fixed`, `predictive`, `short_horizon`, `cem`, and energy-aware guidance
- [controller.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/gnc/controller.py)
  - maps guidance output into TVC, throttle, and roll
- [mission_manager.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/gnc/mission_manager.py)
  - manages launch, ascent, intercept, and failsafe phases
- [safety_guard.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/gnc/safety_guard.py)
  - clips invalid values and prevents NaN or out-of-range actions

## 3. Strategy layer

Main directory:

- [rocket_auto_research/strategies](/F:/Documents/GitHub/Rocket/rocket_auto_research/strategies)

Available strategy families include:

- `baseline_pid`
- `greedy_intercept`
- `predictive_intercept`
- `score_based`
- `mpc_light`
- `cem_planner`
- `energy_aware`
- `rl_policy_wrapper`
- `modular_strategy`

Registry:

- [registry.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/strategies/registry.py)

## 4. Auto Research layer

Main directory:

- [rocket_auto_research/auto_research](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research)

This is the core research engine.

### Problem definition

- [problem_definition.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/problem_definition.py)

Defines internal staged objectives:

- `launch_valid`
- `ascent_stable`
- `energy_margin`
- `approach_window`
- `balloon_pop`

Each stage has:

- a title
- a description
- a promotion threshold
- preferred strategies
- stage bootstrap hypotheses

### Evaluator

- [evaluator.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/evaluator.py)

Computes:

- `mission_fitness`
- `stage_fitness`
- `stage_success_rate`
- `crash_rate`
- `nan_rate`
- `score_std`
- `mean_popped`
- `mean_min_distance`
- `overall_progress`

Official reward remains a single mission score. Internally, the system uses stage-specific dense objectives to make early optimization possible.

### Failure analysis and hypotheses

- [failure_analyzer.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/failure_analyzer.py)
- [hypothesis_generator.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/hypothesis_generator.py)

Failure types include:

- `crash`
- `altitude_shortfall`
- `velocity_shortfall`
- `wrong_target_or_timing`
- `late_launch`
- `near_miss`
- `target_chattering`
- `wind_drift`
- `sensor_jitter`

Hypothesis generation is heuristic and template-driven. It is not yet a fully autonomous method designer.

### Memory and planning

- [research_memory.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/research_memory.py)
- [next_step_planner.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/next_step_planner.py)

The planner decides:

- current stage
- preferred strategies
- adapter profile
- stage bootstrap specs
- whether to stay or advance
- whether the search is plateauing
- whether to enter stuck mode
- whether to fall back to an earlier stage

### Plateau handling

The system now includes:

- plateau detection
- stuck-mode diversification
- stage fallback

When a stage stalls, the planner can:

- increase mutation scale
- inject more random candidates
- reduce reliance on elites
- fall back to a previous stage if the plateau is crash-heavy

### Experiment execution

- [experiment_runner.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/experiment_runner.py)
- [researcher_loop.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/researcher_loop.py)
- [mutation_engine.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/mutation_engine.py)
- [strategy_crossover.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/strategy_crossover.py)

The main loop performs:

1. stage selection
2. population construction
3. experiment execution
4. evaluation
5. failure analysis
6. hypothesis generation
7. mutation and crossover
8. memory update
9. reporting
10. next-step planning

## Simulation backends

### 1. Competition platform surrogate

- [competition_platform_api.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/competition_platform_api.py)
- [competition_platform_adapter.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/competition_platform_adapter.py)

Purpose:

- fast step-based research loop
- official-like observation and action contract
- controllable wind, noise, and balloon release behavior

This is a surrogate, not the official private runtime.

### 2. ActiveRocketPy adapter

- [activerocketpy_adapter.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/activerocketpy_adapter.py)

Purpose:

- higher-fidelity 6-DoF cross-validation
- real flight dynamics integration via ActiveRocketPy

### 3. Official BalloonPoppingChallenge adapter

- [balloon_challenge_adapter.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/balloon_challenge_adapter.py)
- [balloon_challenge_loader.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/balloon_challenge_loader.py)

Purpose:

- run the real `BalloonPoppingChallenge` Gym environment from the official repository
- evaluate strategies against official observation and action schemas

Important limitation:

The research framework can run the official environment, but the current best strategy is still not competitive there. At the time of writing, recent official scenario runs are still stuck in `approach_window` with `mean_popped = 0` and `crash_rate = 1.0`.

### 4. Adapter routing

- [adapter_factory.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/auto_research/adapter_factory.py)

The main configuration uses a stage router so different stages can target different backends.

## Main configuration

There is now one primary Auto Research configuration:

- [configs/auto_research.yaml](/F:/Documents/GitHub/Rocket/configs/auto_research.yaml)

This is the intended main entrypoint for Auto Research.

It defines:

- adapter routing
- stage policy
- preferred strategies by stage
- global parameter space
- population size
- seeds per experiment
- parallel worker settings
- continuous research mode

Older research-loop configs were moved to:

- [configs/legacy](/F:/Documents/GitHub/Rocket/configs/legacy)

Single fixed-strategy run configs remain in:

- [configs](/F:/Documents/GitHub/Rocket/configs)

Examples:

- [configs/energy_aware.yaml](/F:/Documents/GitHub/Rocket/configs/energy_aware.yaml)
- [configs/energy_transfer_robust.yaml](/F:/Documents/GitHub/Rocket/configs/energy_transfer_robust.yaml)
- [configs/rl_policy_wrapper_mlp.yaml](/F:/Documents/GitHub/Rocket/configs/rl_policy_wrapper_mlp.yaml)

## Parallel execution

The system supports CPU parallelism where safe.

Currently:

- safe for parallel execution
  - `mock`
  - `competition_platform`
- kept serial for stability
  - `balloon_challenge`
  - `activerocketpy`

The runtime reports:

- configured workers
- active workers
- worker mode

## Dashboard and replay

Main components:

- [rocket_auto_research/dashboard_builder.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/dashboard_builder.py)
- [rocket_auto_research/dashboard_server.py](/F:/Documents/GitHub/Rocket/rocket_auto_research/dashboard_server.py)
- [experiments/build_dashboard.py](/F:/Documents/GitHub/Rocket/experiments/build_dashboard.py)
- [experiments/run_dashboard_server.py](/F:/Documents/GitHub/Rocket/experiments/run_dashboard_server.py)

Features:

- bilingual dashboard
- day and night themes
- start, pause, resume, and stop research
- promotion progress bar
- stage pipeline
- failure breakdown
- generation trends
- worker status
- 3D replay with fixed scale and manual scale control

Convenience scripts:

- [start.cmd](/F:/Documents/GitHub/Rocket/start.cmd)
- [stop.cmd](/F:/Documents/GitHub/Rocket/stop.cmd)
- [clean_result.cmd](/F:/Documents/GitHub/Rocket/clean_result.cmd)

## Learning utilities

Main directory:

- [rocket_auto_research/learning](/F:/Documents/GitHub/Rocket/rocket_auto_research/learning)

Available workflows:

- behavioral cloning
- MLP policy training
- DAgger iteration
- imitation dataset collection

Scripts:

- [experiments/collect_imitation_dataset.py](/F:/Documents/GitHub/Rocket/experiments/collect_imitation_dataset.py)
- [experiments/train_behavioral_cloning.py](/F:/Documents/GitHub/Rocket/experiments/train_behavioral_cloning.py)
- [experiments/run_dagger_iteration.py](/F:/Documents/GitHub/Rocket/experiments/run_dagger_iteration.py)

## Results layout

Generated outputs live under [results](/F:/Documents/GitHub/Rocket/results):

- [results/runs](/F:/Documents/GitHub/Rocket/results/runs)
  - per-experiment specs, summaries, failure reports, and trajectories
- [results/leaderboards](/F:/Documents/GitHub/Rocket/results/leaderboards)
  - ranked generation outputs
- [results/reports](/F:/Documents/GitHub/Rocket/results/reports)
  - generation-level markdown reports
- [results/research_memory](/F:/Documents/GitHub/Rocket/results/research_memory)
  - planner state and memory artifacts
- [results/best_agents](/F:/Documents/GitHub/Rocket/results/best_agents)
  - `best_config.json` and snapshots
- [results/dashboard](/F:/Documents/GitHub/Rocket/results/dashboard)
  - generated dashboard assets
- [results/models](/F:/Documents/GitHub/Rocket/results/models)
  - learned policy weights
- [results/datasets](/F:/Documents/GitHub/Rocket/results/datasets)
  - imitation and DAgger datasets

## Official BalloonPoppingChallenge compatibility

Official repository:

- [ARRC-Rocket/BalloonPoppingChallenge](https://github.com/ARRC-Rocket/BalloonPoppingChallenge)

What is aligned already:

- official-like observation schema
- official-like action schema
- official `BalloonPoppingChallenge` Gym environment adapter
- scenario loading from official challenge files

What is not yet complete:

- a submission-ready official `agents/agent.py`
- a wrapper that inherits the official `BaseAgent`
- a validated final submission package for the official evaluation path

This matters because the official repo expects:

- an agent in the official `agents` directory
- a class inheriting `BaseAgent`
- a `get_action(observation)` method

The repository can already research against the official environment. It does not yet produce the final official agent wrapper automatically.

## Quick start

### Install

```bash
python -m pip install -e .
```

### Run one fixed strategy

```bash
python experiments/run_single.py --config configs/energy_aware.yaml --seed 0
```

### Run a batch

```bash
python experiments/run_batch.py --config configs/energy_aware.yaml --seeds 0 1 2 3 4
```

### Run Auto Research

```bash
python experiments/run_research_loop.py --config configs/auto_research.yaml
```

### Build the dashboard

```bash
python experiments/build_dashboard.py --results-dir results --output-dir results/dashboard
```

### Run the dashboard server

```bash
python experiments/run_dashboard_server.py
```

Open:

- [http://127.0.0.1:8765](http://127.0.0.1:8765)

### Windows shortcuts

```bat
start.cmd
stop.cmd
clean_result.cmd
```

### Run tests

```bash
python -m unittest discover -s tests -v
```

## Practical limitations

These are the current engineering limits of the project:

- stage definitions are still human-authored
- hypothesis generation is still heuristic
- surrogate-to-official transfer remains a real risk
- the official environment path is slower and less parallel than the surrogate path
- the current best official-env result is still not strong enough for submission
- the repository does not yet export a final official `BaseAgent` implementation automatically

## Recommended next step

If the goal is official competition submission, the next high-value tasks are:

1. build an official submission wrapper that inherits `BaseAgent`
2. keep long-running staged research focused on `balloon_challenge`
3. validate that the new plateau and fallback logic actually breaks the current `approach_window` stall
4. only then freeze a final strategy for submission packaging
