# 🤖 ACE Framework for AppWorld

The **Agent Context Engineering (ACE)** framework implements the learning architecture described in the research paper, utilizing the **AppWorld** dataset for autonomous task execution and adaptation.

This project supports three primary operational modes:

1. **Offline Adaptation (Massive Training Loop)** – Builds a robust, generalized knowledge base.
2. **Online Adaptation (Per-Task Adaptation Loop)** – Dynamically learns and recovers from task failures via self-reflection.
3. **Thompson Sampling-Enhanced Online Adaptation** – Intelligent retry decisions using Bayesian learning.

---

## ✨ Framework Overview

The ACE framework is composed of three key components that work together in a continuous learning loop:

### Core Components

* **Agent (Generator)** – The ReAct-style agent responsible for generating initial action plans and executing tasks.
* **Reflector (Critic)** – The self-correction mechanism that analyzes failed trajectories and proposes improvements.
* **Curator (Executor)** – Manages task execution and updates the central playbook with new insights.

### The Playbook: Central Knowledge Repository

The **Playbook** (`playbook.json`) serves as the framework's memory system, containing:
- **Strategies and Hard Rules**: Core principles for task execution
- **API Usage Patterns**: Specific guidance on how to use various APIs
- **Common Mistakes**: Known pitfalls and how to avoid them
- **Verification Checklists**: Steps to ensure correctness
- **Domain Concepts**: Task-specific knowledge
- **Code Patterns**: Reusable code templates

Each playbook entry includes:
- A unique ID for tracking
- Content describing the insight
- Helpful/harmful counters that track effectiveness over time

---

## 🎯 Thompson Sampling Reward Model

### Why Thompson Sampling?

In the original ACE paper, retry decisions were made using simple heuristics. However, this approach has limitations:

1. **No Learning**: Heuristics don't improve with experience
2. **Fixed Thresholds**: Can't adapt to different task distributions
3. **No Component Attribution**: Can't identify which components (Playbook, Reflector, Curator) are actually helping
4. **Exploration vs Exploitation**: Can't balance trying new strategies vs using proven ones

The **Thompson Sampling** enhancement addresses these issues by implementing **principled Bayesian learning** for retry decisions.

### How Thompson Sampling Works

Thompson Sampling is a Bayesian approach that:

1. **Maintains Beliefs**: Each component (Playbook, Reflector, Curator) has a reliability score modeled as a Beta distribution `Beta(α, β)`
   - Higher α → component is more reliable
   - Higher β → component is less reliable
   - Mean reliability = α / (α + β)

2. **Samples for Decisions**: When deciding whether to retry:
   - Sample reliability scores from each component's distribution
   - Compute expected probability of success: `P(success) = q_R × θ_R + q_C × θ_C`
     - q_R = reflection quality (0-1)
     - q_C = curation value (0-1)
     - θ_R, θ_C = sampled component reliabilities
   - Calculate expected value: `EV = P(success) × reward - attempt_cost`
   - **Retry if EV > 0**

3. **Updates from Outcomes**: After each task:
   - **Success**: Increase α for involved components (credit attribution)
   - **Failure**: Increase β for all components (shared blame)
   - Components naturally converge to their true reliability over time

### Reward Structure

The reward model implements a **decreasing returns** schedule:

```
Attempt 1: +1.0 reward (if successful)
Attempt 2: +0.6 reward (if successful)
Attempt 3: +0.3 reward (if successful)
Cost per attempt: -0.1 × attempt_number
```

This encourages:
- Quick success (highest reward on first attempt)
- Economical retries (diminishing returns, increasing costs)
- Learning when to give up (negative EV → skip retry)

### Quality-Weighted Credit Attribution

When a task succeeds, we need to assign credit to the components that helped. The system uses **quality-weighted attribution**:

```python
# Playbook contribution decreases over attempts
playbook_weight = {1: 1.0, 2: 0.2, 3: 0.1}[attempt]

# Other components based on quality scores
reflector_weight = reflection_quality  # 0-1
curator_weight = curation_value        # 0-1

# Normalize to sum to 1.0
total = playbook_weight + reflector_weight + curator_weight
final_weights = {comp: weight/total for comp, weight in ...}

# Update beliefs proportionally
playbook.alpha += final_weights['playbook']
reflector.alpha += final_weights['reflector']
curator.alpha += final_weights['curator']
```

This ensures:
- Early success credits the Playbook (it already had good knowledge)
- Later success credits the Reflector and Curator (they added new insights)
- High-quality reflections/curations get more credit

---

## 🔄 Code Flow: Thompson Sampling Integration

### 1. Initialization

```
ThompsonSamplingPolicy
├── playbook: Beta(1, 1)      # Uniform prior
├── reflector: Beta(1, 1)     # No initial bias
├── curator: Beta(1, 1)       # Learns over time
└── attempt_rewards: {1: 1.0, 2: 0.6, 3: 0.3}
```

### 2. Single Task Flow (`run_online_thompson.py`)

```
For each attempt (1 to MAX_ONLINE_RETRIES):
    ┌─────────────────────────────────────┐
    │ 1. GENERATE                         │
    │    - Load playbook                  │
    │    - Execute task with Agent        │
    │    - Calculate attempt reward       │
    └─────────────────────────────────────┘
              ↓
    ┌─────────────────────────────────────┐
    │ 2. CHECK SUCCESS                    │
    └─────────────────────────────────────┘
              ↓
         [Success?]
              ↓
         Yes ↓    No →
              ↓         ↓
    ┌─────────────────────────────────────┐
    │ Update beliefs with SUCCESS         │
    │ - Credit attribution based on       │
    │   attempt # and quality scores      │
    │ - Increase α for contributing       │
    │   components                        │
    └─────────────────────────────────────┘
              ↓
         [DONE]
              
              ↓
    ┌─────────────────────────────────────┐
    │ 3. REFLECT                          │
    │    - Analyze trajectory             │
    │    - Calculate reflection_quality   │
    └─────────────────────────────────────┘
              ↓
    ┌─────────────────────────────────────┐
    │ 4. CURATE                           │
    │    - Generate playbook updates      │
    │    - Calculate curation_value       │
    └─────────────────────────────────────┘
              ↓
    ┌─────────────────────────────────────┐
    │ 5. THOMPSON SAMPLING DECISION       │
    │    - Sample θ_R, θ_C from beliefs   │
    │    - Compute P(success)             │
    │    - Calculate EV                   │
    │    - Decide: RETRY or SKIP          │
    └─────────────────────────────────────┘
              ↓
         [Should retry?]
              ↓
         Yes ↓    No →
              ↓         ↓
    ┌─────────────────────────────────────┐
    │ Apply playbook updates              │
    │ Continue to next attempt            │
    └─────────────────────────────────────┘
              
              ↓
    ┌─────────────────────────────────────┐
    │ Update beliefs with FAILURE         │
    │ - Increase β for all components     │
    └─────────────────────────────────────┘
              ↓
         [DONE - Failed]
```

### 3. Batch Learning Mode

When processing multiple tasks with a **shared policy**, the system learns across tasks:

```
tasks = [task1, task2, ..., taskN]
shared_policy = ThompsonSamplingPolicy()

For each task in tasks:
    success, reward = run_online_loop_with_thompson_sampling(
        task, thompson_policy=shared_policy
    )
    
    # Policy learns from this outcome
    # Next task benefits from learned beliefs
```

This enables:
- **Transfer learning**: Knowledge from task1 helps with task2
- **Faster convergence**: Beliefs improve with each task
- **Better decisions**: Policy gets smarter over time

---

## 📊 Key Differences: Heuristic vs Thompson Sampling

| Aspect | Heuristic Approach | Thompson Sampling |
|--------|-------------------|-------------------|
| **Decision Making** | Fixed thresholds | Learned from experience |
| **Component Learning** | No component tracking | Beta distributions per component |
| **Exploration** | None (always greedy) | Balances explore/exploit |
| **Adaptation** | Static rules | Adapts to task distribution |
| **Credit Assignment** | No attribution | Quality-weighted credit |
| **Convergence** | N/A | Provably optimal |
| **Uncertainty** | Ignored | Explicitly modeled (variance) |

---

## 📁 Project Structure

| File/Directory                        | Role          | Description                                                      |
| ------------------------------------- | ------------- | ---------------------------------------------------------------- |
| `run_offline.py`                      | Execution     | **➡️ Run this for Offline Adaptation (Algorithm 1)**             |
| `run_online.py`                       | Execution     | **➡️ Run this for Online Adaptation (Algorithm 2)**              |
| `run_online_thompson.py`              | Execution     | **➡️ Run this for Thompson Sampling-Enhanced Online Adaptation** |
| `run_react.py`                        | Execution     | Simple single-task execution without adaptation                  |
| `playbook.json`                       | Knowledge     | Central knowledge base (initially `{}`)                          |
| `requirements.txt`                    | Dependencies  | Lists required Python packages                                   |
| `.env`                                | Configuration | *(Create this)* – Stores environment and API keys                |
| `ace_appworld/`                       | Source Code   | Main Python package containing core logic                        |
| `ace_appworld/config.py`              | Configuration | Defines parameters like `MAX_ONLINE_RETRIES`, etc.               |
| `ace_appworld/components/`            | Components    | Core modules – `agent.py`, `reflector.py`, `curator.py`          |
| `ace_appworld/components/thompson_sampling.py` | Learning | Thompson Sampling policy implementation             |
| `ace_appworld/components/online_reward_tracker.py` | Learning | Reward tracking and decision logic          |
| `ace_appworld/utils/`                 | Utilities     | Helper functions, including `appworld_loader.py`                 |

---

## ⚙️ Setup & Installation

We recommend using **uv** for fast environment setup and dependency management.

### 1. Install uv and Dependencies

```bash
# Install uv if not available
pip install uv

# Create and activate a Python 3.11 environment
uv venv --python 3.11.8
source .venv/bin/activate

# Install project dependencies
uv pip install -r requirements.txt
uv pip install --upgrade typer click
```

### 2. Prepare Data and Configuration

You must install the **AppWorld** data locally and configure your API keys.

```bash
# Install AppWorld data utilities and download dataset
appworld install
appworld download data

# 🔑 Set API Keys
# Create .env file
# Add LLM and API keys manually inside .env
```

---

## 🏃 How to Run the Framework

### 1. Offline Adaptation – *Algorithm 1: Sequential Training*

Iterates through all **13,101 tasks** in the AppWorld training set, updating `playbook.json` after each task to build a large-scale knowledge base.

> ⚠️ **Caution:** This is a long-running process meant for creating the master playbook.

```bash
uv run run_offline.py

# Optional: Start from a specific task index
uv run run_offline.py --start_from 100
```

### 2. Online Adaptation – *Algorithm 2: Single Task Retry*

Runs a specific task using the existing playbook. If the task fails, the Reflector critiques it, updates the playbook, and retries until success or until the retry limit is reached.

```bash
uv run run_online.py <task_id>

# Example:
uv run run_online.py b0a8eae_3
```

### 3. Thompson Sampling-Enhanced Online Adaptation – *Intelligent Retry Decisions*

Uses Bayesian learning to make smarter retry decisions based on component reliability and expected value.

#### Single Task Mode

```bash
uv run run_online_thompson.py <task_id>

# Example:
uv run run_online_thompson.py b0a8eae_3
```

#### Multiple Tasks (Batch Mode with Shared Learning)

```bash
# Process multiple tasks with shared learning
uv run run_online_thompson.py task1 task2 task3

# Or from a file (one task per line)
uv run run_online_thompson.py --batch task_list.txt
```

**Batch mode benefits:**
- Policy learns from each task
- Beliefs improve over time
- Later tasks benefit from earlier learning
- Results saved to `thompson_batch_results.json`

#### Understanding the Output

```
========================================
Thompson Sampling Decision (attempt 2)
  Quality scores: q_R=0.750, q_C=0.600
  Sampled beliefs: theta_R=0.623, theta_C=0.544
  P(success) = 0.794
  Expected reward = 0.476, Cost = 0.200
  Net EV = +0.276
  Decision: RETRY
========================================

# After task completion:
Thompson Sampling Summary
==========================
Total tasks: 10
Success rate: 9/10

Learned component reliabilities:
  Playbook:
    Mean reliability: 0.804
    Uncertainty (var): 0.014
    Beta(8.22, 2.00)
    Based on 8 observations

  Reflector:
    Mean reliability: 0.716
    Uncertainty (var): 0.0449
    Beta(2.52, 1.00)
    Based on 2 observations

  Curator:
    Mean reliability: 0.557
    Uncertainty (var): 0.0758
    Beta(1.26, 1.00)
    Based on 0 observations
```

### 4. Simple React Agent – *No Adaptation*

Runs a single task once without any learning or retry logic.

```bash
uv run run_react.py <task_id>

# Example:
uv run run_react.py b0a8eae_3
```

---

## 🧠 ACE Framework Deep Dive

### The Learning Cycle

The ACE framework implements a continuous learning cycle:

```
           ┌──────────────┐
           │   PLAYBOOK   │ (P)
           │  (Knowledge) │
           └───────┬──────┘
                   │
                   ↓ provides guidance
           ┌──────────────┐
           │    AGENT     │ (Generator)
           │  (M_gen)     │
           └───────┬──────┘
                   │
                   ↓ produces
           ┌──────────────┐
           │  TRAJECTORY  │ (τ)
           │  (execution) │
           └───────┬──────┘
                   │
                   ↓ analyzed by
           ┌──────────────┐
           │  REFLECTOR   │ (Critic)
           │  (M_ref)     │
           └───────┬──────┘
                   │
                   ↓ generates
           ┌──────────────┐
           │  REFLECTION  │ (r)
           │  (insights)  │
           └───────┬──────┘
                   │
                   ↓ curated by
           ┌──────────────┐
           │   CURATOR    │ (Executor)
           │  (M_cur)     │
           └───────┬──────┘
                   │
                   ↓ updates
           ┌──────────────┐
           │  PLAYBOOK'   │ (P')
           └──────────────┘
                   │
                   └──→ (cycle continues)
```

### Component Interactions

1. **Agent ← Playbook**: Agent reads playbook entries to guide task execution
2. **Agent → Trajectory**: Agent produces step-by-step execution trace
3. **Reflector ← Trajectory + Ground Truth**: Analyzes what went wrong
4. **Reflector → Reflection**: Generates insights and bullet tags
5. **Curator ← Reflection + Playbook**: Determines what to add/update
6. **Curator → Operations**: Proposes ADD operations for playbook
7. **PlaybookManager ← Operations**: Applies changes, deduplicates, prunes
8. **Thompson Sampling**: Decides if retry is worthwhile based on learned beliefs

### Offline vs Online Learning

**Offline Adaptation (Algorithm 1)**:
- Processes all 13,101 tasks sequentially
- One attempt per task (no retries)
- Builds comprehensive playbook
- Good for: Initial knowledge acquisition
- Time: Days/weeks for full dataset

**Online Adaptation (Algorithm 2)**:
- Focuses on single task
- Multiple retry attempts
- Task-specific learning
- Good for: Deployment, edge cases
- Time: Minutes per task

**Thompson Sampling Mode**:
- Combines benefits of both
- Learns optimal retry policy
- Batch mode enables transfer learning
- Good for: Efficient online deployment
- Time: Minutes per task, improves over time

---

## 📖 Related Resources

| Resource                  | Description                              | Link                                                                |
| ------------------------- | ---------------------------------------- | ------------------------------------------------------------------- |
| **ACE Paper**             | Core framework and algorithms            | [arXiv:2510.04618](https://arxiv.org/pdf/2510.04618)               |
| **AppWorld Dataset**      | Project repository and installation info | [GitHub: AppWorld](https://github.com/stonybrooknlp/appworld/)      |
| **AppWorld Paper**        | Dataset and benchmark details            | [arXiv:2407.18901](https://arxiv.org/pdf/2407.18901)               |
| **Thompson Sampling**     | Original algorithm                       | [Tutorial](https://en.wikipedia.org/wiki/Thompson_sampling)         |
| **Beta Distribution**     | Used for belief modeling                 | [Wikipedia](https://en.wikipedia.org/wiki/Beta_distribution)        |

---

## 🔍 Monitoring and Debugging

### Key Metrics to Track

1. **Success Rate**: Tasks completed successfully
2. **Average Attempts**: How many retries before success/failure
3. **Cumulative Reward**: Total reward across tasks
4. **Component Reliabilities**: Alpha/Beta parameters over time
5. **Decision Quality**: EV calculations, retry decisions

### Log Files

- Main log: `outputs/logs/ace_appworld.log`
- Episode traces: `outputs/traces/`
- Reflections: `outputs/reflections/`
- AppWorld evaluations: `experiments/outputs/`

### Troubleshooting Thompson Sampling

**Problem**: Policy always skips retries
- **Check**: Initial beliefs may be too pessimistic
- **Solution**: Adjust priors in `ThompsonSamplingPolicy.__init__()`

**Problem**: Too many retries on obviously bad tasks
- **Check**: Reward structure may be too generous
- **Solution**: Increase `cost_per_attempt` or decrease attempt rewards

**Problem**: Not learning from successful tasks
- **Check**: Credit attribution weights
- **Solution**: Review `calculate_credit_weights()` logic

---

