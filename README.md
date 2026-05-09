# 火箭自動研究系統
# Rocket Auto Research

這個專案不是要手寫一個固定火箭控制器，而是要建立一套可持續演化的火箭 GNC 自動研究系統。
This project is not about hand-writing one fixed rocket controller; it is about building an evolving Auto Research system for rocket GNC.

系統會自動產生策略、批次跑模擬、評估結果、記錄研究記憶，並根據失敗型態決定下一步研究方向。
The system automatically generates strategies, runs batched simulations, evaluates results, records research memory, and decides the next research direction from failure patterns.

## 核心定位
## Core Positioning

不是寫一個火箭控制器。
This is not just writing a rocket controller.

而是寫一個會持續找出更好控制器的研究器。
It is building a researcher that keeps finding better controllers.

## 專案目前包含什麼
## What This Repository Contains

- staged Auto Research 主循環
- staged Auto Research main loop
- 共用 GNC 模組
- shared GNC modules
- 多種策略家族
- multiple strategy families
- surrogate 研究環境
- surrogate research environment
- 官方 `BalloonPoppingChallenge` 環境 adapter
- official `BalloonPoppingChallenge` environment adapter
- ActiveRocketPy 高保真 cross-validation adapter
- ActiveRocketPy high-fidelity cross-validation adapter
- HTML dashboard、3D replay、執行控制
- HTML dashboard, 3D replay, and runtime control
- imitation learning 與 DAgger 工具
- imitation learning and DAgger utilities

## 專案刻意不放進版控的內容
## What Is Intentionally Not Tracked

這個 repo 已經瘦身，不把大型外部依賴與生成結果直接放進 git。
This repo is intentionally slim and does not track large external dependencies or generated outputs in git.

- `.external/BalloonPoppingChallenge/`
- `.external/BalloonPoppingChallenge/`
- `vendor/`
- `vendor/`
- `results/`
- `results/`

## 專案結構
## Repository Layout

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

## 外部依賴安裝
## External Dependency Bootstrap

官方 challenge 與 ActiveRocketPy 不是直接放在 repo 內，而是按需抓取。
The official challenge and ActiveRocketPy are not vendored in the repo and are fetched on demand.

可以用下面任一方式 bootstrap。
You can bootstrap them with either of the following commands.

```bat
bootstrap_external.cmd
```

```bash
python experiments/bootstrap_external_dependencies.py
```

這個腳本會做兩件事。
That script performs two tasks.

1. 把 `ARRC-Rocket/BalloonPoppingChallenge` clone 到 `.external/BalloonPoppingChallenge`
1. Clone `ARRC-Rocket/BalloonPoppingChallenge` into `.external/BalloonPoppingChallenge`
2. 初始化其 `ActiveRocketPy` submodule
2. Initialize its `ActiveRocketPy` submodule

完成後預期本機會有以下路徑。
After bootstrap, these local paths are expected to exist.

- `.external/BalloonPoppingChallenge`
- `.external/BalloonPoppingChallenge`
- `.external/BalloonPoppingChallenge/ActiveRocketPy`
- `.external/BalloonPoppingChallenge/ActiveRocketPy`

## 系統分層架構
## Layered Architecture

### Agent 層
### Agent Layer

主要入口在 `rocket_auto_research/agents/competition_agent.py`。
The main entrypoint is `rocket_auto_research/agents/competition_agent.py`.

職責如下。
Its responsibilities are listed below.

- 驗證 observation payload
- validate observation payloads
- 解析成內部 `WorldState`
- parse them into internal `WorldState`
- 呼叫目前選定策略
- call the currently selected strategy
- 格式化並驗證 action
- format and validate actions

這是 repo 內部的 agent 邊界，還不是官方最終提交 wrapper。
This is the repository's internal agent boundary, not yet the final official submission wrapper.

### GNC 層
### GNC Layer

共用 GNC 模組位於 `rocket_auto_research/gnc/`。
Shared GNC modules live under `rocket_auto_research/gnc/`.

主要模組如下。
The key modules are listed below.

- `observation_parser.py`
- `observation_parser.py`
- `state.py`
- `state.py`
- `estimator.py`
- `estimator.py`
- `target_selector.py`
- `target_selector.py`
- `guidance.py`
- `guidance.py`
- `controller.py`
- `controller.py`
- `mission_manager.py`
- `mission_manager.py`
- `safety_guard.py`
- `safety_guard.py`

### 策略層
### Strategy Layer

策略家族位於 `rocket_auto_research/strategies/`。
Strategy families live under `rocket_auto_research/strategies/`.

目前可用策略包含以下幾類。
The currently available strategies include the following families.

- `baseline_pid`
- `baseline_pid`
- `greedy_intercept`
- `greedy_intercept`
- `predictive_intercept`
- `predictive_intercept`
- `score_based`
- `score_based`
- `mpc_light`
- `mpc_light`
- `cem_planner`
- `cem_planner`
- `energy_aware`
- `energy_aware`
- `rl_policy_wrapper`
- `rl_policy_wrapper`
- `modular_strategy`
- `modular_strategy`

### Auto Research 層
### Auto Research Layer

核心研究模組位於 `rocket_auto_research/auto_research/`。
The core research modules live under `rocket_auto_research/auto_research/`.

主要元件如下。
The main components are listed below.

- `problem_definition.py`
- `problem_definition.py`
- `evaluator.py`
- `evaluator.py`
- `failure_analyzer.py`
- `failure_analyzer.py`
- `hypothesis_generator.py`
- `hypothesis_generator.py`
- `research_memory.py`
- `research_memory.py`
- `next_step_planner.py`
- `next_step_planner.py`
- `mutation_engine.py`
- `mutation_engine.py`
- `researcher_loop.py`
- `researcher_loop.py`

## staged 自動研究
## Staged Auto Research

系統不直接只看官方單一分數，而是先拆成內部 stage 逐步推進。
The system does not optimize only the single official reward directly; it first decomposes the problem into internal stages and advances step by step.

目前 stage 順序如下。
The current stage sequence is listed below.

- `launch_valid`
- `launch_valid`
- `ascent_stable`
- `ascent_stable`
- `energy_margin`
- `energy_margin`
- `approach_window`
- `approach_window`
- `balloon_pop`
- `balloon_pop`

官方任務分數仍然只有一個，但內部 evaluator 會對每個 stage 定義較密集的成功條件與 fitness。
The official task still has a single score, but the internal evaluator defines denser success conditions and fitness for each stage.

## 失敗後如何調整
## How The System Adjusts After Failure

研究流程不是只做參數掃描。
The research flow is not just parameter sweeping.

每一代會做以下事情。
Each generation performs the following steps.

1. 跑完整個 population
1. Run the full population
2. 分析主要 failure pattern
2. Analyze dominant failure patterns
3. 產生 hypothesis
3. Generate hypotheses
4. 保留 elites
4. Keep elites
5. 做 mutation、crossover、random injection
5. Apply mutation, crossover, and random injection
6. 規劃下一代 stage 與研究策略
6. Plan the next generation stage and research strategy

## 脫困機制
## Plateau Escape Mechanisms

當系統長時間卡住，不會只無限做小幅調參。
When the system is stuck for a long time, it no longer just performs endless small parameter tweaks.

目前已經實作以下三種脫困機制。
The following three escape mechanisms are now implemented.

### 1. `approach_window` 專屬 hard-reset 模式
### 1. `approach_window`-specific hard-reset mode

如果 `approach_window` 連續 plateau，planner 會進入 hard-reset。
If `approach_window` repeatedly plateaus, the planner enters hard-reset mode.

這會提高 mutation 強度、提高 random injection 比例，並減少對舊 elites 的依賴。
This increases mutation strength, increases random injection ratio, and reduces dependence on old elites.

### 2. 封鎖最近 top-K 相似參數區域
### 2. Blocking recently repeated top-K similar parameter regions

如果 repeated plateau 發生，系統會從最近 top-K record 建立 blocked regions。
If repeated plateau occurs, the system builds blocked regions from recent top-K records.

後續 mutation 與 random candidate 會盡量避開這些相似參數區域。
Subsequent mutations and random candidates try to avoid these similar parameter regions.

### 3. fallback 後重新進場時的暫時性 boost
### 3. Temporary boost after fallback and re-entry

如果 `approach_window` crash-heavy plateau 後 fallback 到前一 stage，再重新進場時會暫時提高探索預算。
If `approach_window` falls back because of a crash-heavy plateau, the next re-entry temporarily increases exploration budget.

目前 boost 會暫時提高以下兩項。
The boost currently increases the following items temporarily.

- `population size`
- `population size`
- `seeds_per_experiment`
- `seeds_per_experiment`

這樣 re-entry 時不是用原本同樣的狹窄設定再撞一次。
This prevents re-entry from immediately retrying the same narrow setup.

## 模擬後端
## Simulation Backends

### surrogate 研究後端
### Surrogate Research Backend

主要檔案如下。
The main files are listed below.

- `competition_platform_api.py`
- `competition_platform_api.py`
- `competition_platform_adapter.py`
- `competition_platform_adapter.py`

用途如下。
It is used for the following purposes.

- 快速迭代
- fast iteration
- stage bootstrapping
- stage bootstrapping
- 安全的 CPU 平行執行
- safe CPU parallel execution

### 官方 challenge 後端
### Official Challenge Backend

主要檔案如下。
The main files are listed below.

- `balloon_challenge_adapter.py`
- `balloon_challenge_adapter.py`
- `balloon_challenge_loader.py`
- `balloon_challenge_loader.py`

用途如下。
It is used for the following purposes.

- 真實 `BalloonPoppingChallenge` Gym 評估
- real `BalloonPoppingChallenge` Gym evaluation
- 官方 observation / action contract 研究
- research on the official observation / action contract
- 官方環境下的 CPU 平行執行
- CPU parallel execution on the official environment

官方環境本身會在其 package tree 內寫 cache。
The official environment writes cache data inside its package tree.

為了安全平行執行，這個 repo 會在每個 worker 啟動前建立隔離的臨時 runtime copy。
To make parallel execution safe, this repo creates an isolated temporary runtime copy before each worker imports and runs it.

### ActiveRocketPy 高保真後端
### ActiveRocketPy High-Fidelity Backend

主要檔案如下。
The main file is listed below.

- `activerocketpy_adapter.py`
- `activerocketpy_adapter.py`

用途如下。
It is used for the following purposes.

- 高保真 cross-validation
- high-fidelity cross-validation
- 6-DoF 行為檢查
- 6-DoF behavior checks

## 主設定檔
## Main Configuration

唯一主 Auto Research 設定檔如下。
The single main Auto Research configuration is listed below.

- `configs/auto_research.yaml`
- `configs/auto_research.yaml`

舊 loop 設定檔保留在以下位置，只作歷史參考。
Older loop configs are kept only as historical references in the following location.

- `configs/legacy/`
- `configs/legacy/`

單策略固定設定檔仍保留在 `configs/`。
Single-strategy fixed configs are still kept under `configs/`.

## Dashboard
## Dashboard

主要檔案如下。
The main files are listed below.

- `rocket_auto_research/dashboard_builder.py`
- `rocket_auto_research/dashboard_builder.py`
- `rocket_auto_research/dashboard_server.py`
- `rocket_auto_research/dashboard_server.py`
- `experiments/run_dashboard_server.py`
- `experiments/run_dashboard_server.py`

目前 dashboard 已提供以下功能。
The dashboard currently provides the following features.

- 中英雙語介面
- bilingual interface
- 白天 / 黑夜主題
- day / night themes
- start / pause / resume / stop
- start / pause / resume / stop
- worker 數量控制
- worker count control
- population size 控制
- population size control
- stage pipeline 與 promotion progress
- stage pipeline and promotion progress
- failure breakdown
- failure breakdown
- experiment evolution 圖
- experiment evolution chart
- 3D replay 與 timeline
- 3D replay and timeline
- RAM buffer 與 flush progress 顯示
- RAM buffer and flush progress display

### 實驗演進圖
### Experiment Evolution Chart

這張圖回答的是一個很具體的問題。
This chart answers one very specific question.

第 N 次實驗如果分數提升了，當時到底改了什麼。
If experiment N improved the score, what exactly changed at that point.

圖上會顯示以下資訊。
The chart shows the following information.

- discarded experiments
- discarded experiments
- kept improvements
- kept improvements
- running best 階梯線
- running-best staircase
- 每個有效提升點的關鍵變更標註
- key-change annotations for each kept improvement

Y 軸可以切換不同指標，例如 `final_fitness` 或 `popped_count`。
The Y-axis can switch between different metrics such as `final_fitness` or `popped_count`.

## 學習工具
## Learning Utilities

主要腳本如下。
The main scripts are listed below.

- `experiments/collect_imitation_dataset.py`
- `experiments/collect_imitation_dataset.py`
- `experiments/train_behavioral_cloning.py`
- `experiments/train_behavioral_cloning.py`
- `experiments/run_dagger_iteration.py`
- `experiments/run_dagger_iteration.py`

## 建議工作流程
## Typical Workflow

### 1. 安裝套件
### 1. Install the package

```bash
python -m pip install -e .
```

### 2. 抓取外部依賴
### 2. Fetch external dependencies

```bash
python experiments/bootstrap_external_dependencies.py
```

### 3. 執行測試
### 3. Run tests

```bash
python -m unittest discover -s tests -v
```

如果缺少官方外部 repo，對應測試會自動 skip。
If the official external repos are missing, the related tests will skip automatically.

### 4. 啟動 Auto Research
### 4. Start Auto Research

```bash
python experiments/run_research_loop.py --config configs/auto_research.yaml
```

worker 數與 population size 也可以從 dashboard 設定。
Worker count and population size can also be set from the dashboard.

新的設定只會套用在下一輪新啟動的 research run。
New settings apply only to the next newly started research run.

### 5. 啟動 dashboard
### 5. Start the dashboard

```bat
start.cmd
```

### 6. 清除舊結果
### 6. Clean old results

```bat
clean_result.cmd
```

## 公開維護政策
## Public Maintenance Policy

`Learn_Rocket` 這個公開 repo 的維護原則如下。
The maintenance policy for the public `Learn_Rocket` repo is listed below.

- 保留必要原始碼
- keep necessary source code
- 不追蹤研究輸出
- do not track generated research outputs
- 不 vendoring 大型第三方 repo
- do not vendor heavy third-party repos
- 用 bootstrap 步驟取得官方依賴
- fetch official dependencies through bootstrap steps
- 用文件清楚說明 setup
- document setup clearly

## 目前限制
## Current Limitations

- 目前還沒有自動輸出官方 `BaseAgent` 提交 wrapper
- the repo still does not automatically export a final official `BaseAgent` submission wrapper
- 官方 env 的最終策略表現仍弱於研究架構本身的成熟度
- official-env strategy performance is still weaker than the maturity of the research architecture itself
- stage 定義與 hypothesis template 仍然是人工設計，不是完全自我生成
- stage definitions and hypothesis templates are still human-authored, not fully self-derived

## 官方參考
## Official Reference

官方 challenge repo 如下。
The official challenge repository is listed below.

- [ARRC-Rocket/BalloonPoppingChallenge](https://github.com/ARRC-Rocket/BalloonPoppingChallenge)
- [ARRC-Rocket/BalloonPoppingChallenge](https://github.com/ARRC-Rocket/BalloonPoppingChallenge)
