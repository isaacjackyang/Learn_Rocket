# 火箭自動研究系統
# Rocket Auto Research

這個專案用來研究火箭 GNC 策略。系統會自動產生候選策略、執行批次模擬、評估結果、記錄研究記憶，並決定下一輪要探索的方向。
This project is for researching rocket GNC strategies. The system generates candidate strategies, runs batched simulations, evaluates results, stores research memory, and decides what to explore next.

研究目標是持續提高任務分數、穩定性與對未知 seed 的表現。
The research goal is to keep improving mission score, stability, and performance on unseen seeds.

## 專案內容
## What The Repository Contains

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

## 不納入版控的內容
## What Is Not Tracked In Git

這個 repo 不直接追蹤大型外部依賴與研究輸出。
This repo does not directly track large external dependencies or research outputs.

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

官方 challenge 與 ActiveRocketPy 透過 bootstrap 腳本抓取。
The official challenge and ActiveRocketPy are fetched through the bootstrap script.

可用以下方式安裝。
Use either of the following commands.

```bat
bootstrap_external.cmd
```

```bash
python experiments/bootstrap_external_dependencies.py
```

腳本會執行以下工作。
The script performs the following tasks.

1. 把 `ARRC-Rocket/BalloonPoppingChallenge` clone 到 `.external/BalloonPoppingChallenge`
1. Clone `ARRC-Rocket/BalloonPoppingChallenge` into `.external/BalloonPoppingChallenge`
2. 初始化它的 `ActiveRocketPy` submodule
2. Initialize its `ActiveRocketPy` submodule

完成後預期本機存在以下路徑。
After bootstrap, the following local paths are expected to exist.

- `.external/BalloonPoppingChallenge`
- `.external/BalloonPoppingChallenge`
- `.external/BalloonPoppingChallenge/ActiveRocketPy`
- `.external/BalloonPoppingChallenge/ActiveRocketPy`

## 系統架構
## System Architecture

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

這是 repo 內部 agent 介面，還不是官方最終提交 wrapper。
This is the repository's internal agent interface and not yet the final official submission wrapper.

### GNC 層
### GNC Layer

共用 GNC 模組位於 `rocket_auto_research/gnc/`。
Shared GNC modules are located under `rocket_auto_research/gnc/`.

主要模組如下。
The main modules are listed below.

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
Strategy families are located under `rocket_auto_research/strategies/`.

目前包含以下策略。
The current strategy set includes the following.

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
Core research modules are located under `rocket_auto_research/auto_research/`.

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

系統把任務拆成多個內部 stage，先處理合法發射與穩定爬升，再推進到接近目標與實際戳破氣球。
The system decomposes the mission into internal stages. It first handles valid launch and stable ascent, then moves toward target approach and actual balloon popping.

目前 stage 順序如下。
The current stage order is listed below.

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

官方任務分數只有一個。內部 evaluator 會為每個 stage 提供較密集的 success signal 與 stage fitness。
The official task score is a single scalar. The internal evaluator provides denser success signals and stage fitness for each stage.

## 失敗後的調整邏輯
## Failure Adjustment Logic

每一代會進行以下流程。
Each generation follows the steps below.

1. 執行整個 population
1. Run the full population
2. 分析主要 failure pattern
2. Analyze dominant failure patterns
3. 產生 hypothesis
3. Generate hypotheses
4. 保留 elites
4. Keep elites
5. 執行 mutation、crossover、random injection
5. Apply mutation, crossover, and random injection
6. 規劃下一代 stage 與研究方向
6. Plan the next generation stage and research direction

## plateau 脫困機制
## Plateau Escape Mechanisms

系統已實作以下脫困手段。
The system implements the following plateau escape mechanisms.

### `approach_window` hard-reset
### `approach_window` hard-reset

當 `approach_window` 連續 plateau，planner 會提高 mutation 強度、提高 random injection 比例，並降低對舊 elites 的依賴。
When `approach_window` repeatedly plateaus, the planner increases mutation strength, increases the random injection ratio, and reduces dependence on old elites.

### 封鎖最近 top-K 相似參數區域
### Blocking recently repeated top-K similar parameter regions

系統會根據最近 top-K record 建立 blocked regions。後續 mutation 與 random candidate 會盡量避開這些區域。
The system builds blocked regions from recent top-K records. Later mutations and random candidates try to avoid those regions.

### fallback 後重新進場的暫時性 boost
### Temporary re-entry boost after fallback

如果 `approach_window` 因為 crash-heavy plateau 而 fallback，重新進場時會暫時提高探索預算。
If `approach_window` falls back because of a crash-heavy plateau, the next re-entry temporarily increases exploration budget.

目前 boost 會提高以下項目。
The boost currently increases the following items.

- `population size`
- `population size`
- `seeds_per_experiment`
- `seeds_per_experiment`

這段期間 planner 也會暫時忽略舊的 plateau 記錄，先測試新的候選集合。
During this period, the planner temporarily ignores the old plateau record and evaluates a new candidate set first.

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

官方環境會在 package tree 內寫 cache。這個 repo 會在每個 worker 啟動前建立隔離的暫時 runtime copy。
The official environment writes cache data inside its package tree. This repo creates an isolated temporary runtime copy before each worker starts.

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

主 Auto Research 設定檔如下。
The main Auto Research configuration file is listed below.

- `configs/auto_research.yaml`
- `configs/auto_research.yaml`

舊 loop 設定檔保留在 `configs/legacy/`，只作歷史參考。
Older loop configs are kept under `configs/legacy/` for historical reference only.

單策略固定設定檔保留在 `configs/`。
Single-strategy fixed configs remain under `configs/`.

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

dashboard 提供以下功能。
The dashboard provides the following features.

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

這張圖用來看每次實驗是否提升了最佳表現，以及提升時改了什麼。
This chart shows whether each experiment improved the running best and what changed when it did.

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

Y 軸可以切換指標，例如 `final_fitness` 或 `popped_count`。
The Y-axis can switch metrics such as `final_fitness` or `popped_count`.

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
Worker count and population size can also be configured from the dashboard.

新的設定只會套用到下一輪新啟動的 research run。
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

`Learn_Rocket` 公開 repo 的維護原則如下。
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
- stage 定義與 hypothesis template 仍然是人工設計
- stage definitions and hypothesis templates are still human-authored

## 官方參考
## Official Reference

官方 challenge repo 如下。
The official challenge repository is listed below.

- [ARRC-Rocket/BalloonPoppingChallenge](https://github.com/ARRC-Rocket/BalloonPoppingChallenge)
- [ARRC-Rocket/BalloonPoppingChallenge](https://github.com/ARRC-Rocket/BalloonPoppingChallenge)
