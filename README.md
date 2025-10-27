# 🤖 ACE Framework for AppWorld

The **Agent-Critic-Executor (ACE)** framework implements the learning architecture described in the research paper, utilizing the **AppWorld** dataset for autonomous task execution and adaptation.

This project supports two primary operational modes:

1. **Offline Adaptation (Massive Training Loop)** – Builds a robust, generalized knowledge base.
2. **Online Adaptation (Per-Task Adaptation Loop)** – Dynamically learns and recovers from task failures via self-reflection.

---

## ✨ Framework Overview

The ACE framework is composed of three key components:

* **Agent (Generator)** – The ReAct-style agent responsible for generating initial action plans.
* **Reflector (Critic)** – The self-correction mechanism that analyzes failed trajectories and proposes improvements.
* **Curator (Executor)** – Manages task execution and updates the central playbook.

---

## 📁 Project Structure

| File/Directory             | Role          | Description                                             |
| -------------------------- | ------------- | ------------------------------------------------------- |
| `run_offline.py`           | Execution     | **➡️ Run this for Offline Adaptation (Algorithm 1)**    |
| `run_online.py`            | Execution     | **➡️ Run this for Online Adaptation (Algorithm 2)**     |
| `playbook.json`            | Knowledge     | Central knowledge base (initially `{}`)                 |
| `requirements.txt`         | Dependencies  | Lists required Python packages                          |
| `.env`                     | Configuration | *(Create this)* – Stores environment and API keys       |
| `ace_appworld/`            | Source Code   | Main Python package containing core logic               |
| `ace_appworld/config.py`   | Configuration | Defines parameters like `MAX_ONLINE_RETRIES`, etc.      |
| `ace_appworld/components/` | Components    | Core modules – `agent.py`, `reflector.py`, `curator.py` |
| `ace_appworld/utils/`      | Utilities     | Helper functions, including `appworld_loader.py`        |

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
python run_offline.py
```

### 2. Online Adaptation – *Algorithm 2: Single Task Retry*

Runs a specific task using the existing playbook. If the task fails, the Reflector critiques it, updates the playbook, and retries until success or until the retry limit is reached.

```bash
python run_online.py <task_id>

# Example:
python run_online.py b0a8eae_3
```

---

## 📖 Related Resources

| Resource             | Description                              | Link                                                           |
| -------------------- | ---------------------------------------- | -------------------------------------------------------------- |
| **ACE Paper**        | Core framework and algorithms            | [arXiv:2510.04618](https://arxiv.org/pdf/2510.04618)           |
| **AppWorld Dataset** | Project repository and installation info | [GitHub: AppWorld](https://github.com/stonybrooknlp/appworld/) |
| **AppWorld Paper**   | Dataset and benchmark details            | [arXiv:2407.18901](https://arxiv.org/pdf/2407.18901)           |
