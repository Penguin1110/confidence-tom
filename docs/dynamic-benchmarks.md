# Dynamic Agent Benchmarks

This repo originally focused on static QA-style confidence experiments. The next
stage can use dynamic, environment-based agent benchmarks without forcing all of
them into the main project dependency set.

## Selected Benchmarks

| Benchmark                 | Install Path              | Why This Path                                      | Notes                                         |
| ------------------------- | ------------------------- | -------------------------------------------------- | --------------------------------------------- |
| tau-bench                 | editable source clone     | Official setup is source-first                     | Needs model-provider API keys                 |
| Plancraft                 | pip install               | Official PyPI package exists                       | Good low-friction dynamic planning benchmark  |
| BIRD-SQL                  | sparse clone              | No standalone package; lives inside DAMO-ConvAI    | Keep large benchmark assets outside repo core |
| InterCode                 | pip install               | Official PyPI package exists                       | Still needs Docker for environments           |
| AgentBench OS-Interaction | clone only + separate env | Official repo recommends a separate Python 3.9 env | Docker and compose heavy                      |

## Recommended Strategy

Use the current project environment for the light benchmarks:

- Plancraft
- InterCode
- tau-bench

Keep the heavier and more opinionated benchmarks external:

- BIRD-SQL in an external sparse checkout
- AgentBench OS-Interaction in a separate Python 3.9 environment

On Windows, the Docker-heavy benchmarks are more reliable inside WSL2 with Docker
Desktop integration enabled.

## Bootstrap Command

From the repo root, run (default excludes AgentBench):

```bash
c:/Users/admin/Desktop/Experiment/.venv/Scripts/python.exe tools/setup_dynamic_benchmarks.py
```

If you want to include AgentBench explicitly:

```bash
c:/Users/admin/Desktop/Experiment/.venv/Scripts/python.exe tools/setup_dynamic_benchmarks.py --include-agentbench
```

If you only want to fetch the source repos first (still excluding AgentBench):

```bash
c:/Users/admin/Desktop/Experiment/.venv/Scripts/python.exe tools/setup_dynamic_benchmarks.py --benchmarks tau-bench birdsql --clone-only
```

The script writes a machine-readable summary to:

- external/benchmark_setup_status.json

## Smoke Test (Runnable Check)

After setup, run:

```bash
c:/Users/admin/Desktop/Experiment/.venv/Scripts/python.exe tools/smoke_dynamic_benchmarks.py
```

This validates runnable status for:

- tau-bench CLI
- Plancraft gym wrapper
- InterCode import + Docker connectivity
- BIRD-SQL evaluation script compile

Result file:

- external/dynamic_smoke_status.json

## Benchmark-Specific Notes

### tau-bench

- Repo: https://github.com/sierra-research/tau-bench
- Best path: clone and install editable
- Reason: official setup uses `pip install -e .`

### Plancraft

- Repo: https://github.com/gautierdag/plancraft
- Package: `plancraft`
- Best first integration path: `PlancraftGymWrapper`
- Windows note: the current package has a path parsing bug in `recipes.py`; setup script now auto-patches this after install.

### BIRD-SQL

- Repo root: DAMO-ConvAI, benchmark lives under `bird/`
- No clean standalone package
- Treat it as an external benchmark asset plus evaluation scripts

### InterCode

- Package: `intercode-bench`
- Even when installed from PyPI, the SQL and Bash environments rely on Docker

### AgentBench OS-Interaction

- Repo: https://github.com/THUDM/AgentBench
- Recommended Python: 3.9
- Recommended run mode: separate environment plus Docker Compose
- If your goal is specifically OS interaction, you do not need to wire up the full benchmark suite on day one

## Practical Next Step

If you want the lowest-friction path for the next experimental iteration, start
with this order:

1. Plancraft
2. InterCode SQL or Bash
3. tau-bench
4. BIRD-SQL
5. AgentBench OS-Interaction