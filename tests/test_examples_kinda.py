import subprocess
import pytest
import os
import sys
import time

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "../examples")
EXAMPLE_SCRIPTS = [
    "maze_dr.py",
    "maze_plr.py",
    "maze_paired.py",
]

@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS)
def test_run_example(script):
    script_path = os.path.join(EXAMPLES_DIR, script)
    assert os.path.exists(script_path), f"Script {script} not found."

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    try:
        process = subprocess.run(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            env=env,
        )
        assert process.returncode in [None, 0], f"Script {script} failed:\n{process.stderr.decode()}"
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        pytest.fail(f"Error running {script}: {e}")
