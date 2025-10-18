import re
import subprocess
import sys
from pprint import pformat
from typing import List

import typer
from loguru import logger


def main(
    stage: list[str] = typer.Argument(..., help="List of stages to commit"),
    glob: bool = typer.Option(False, "--glob", help="Use glob pattern to match stages"),
) -> None:
    stages_to_commit = get_stages_to_run(stage, glob)

    logger.info(f"Stages to commit:\n{pformat(stages_to_commit)}")

    if not stages_to_commit:
        logger.info("No stages found to commit")
        return

    success_count = 0
    failure_count = 0

    for stage_name in stages_to_commit:
        if commit_stage(stage_name):
            success_count += 1
        else:
            failure_count += 1

    logger.info(f"Commit summary: {success_count} successful, {failure_count} failed")


def get_stages_to_run(stages: list[str], glob: bool) -> List[str]:
    """Run dvc repro --dry and extract stages that would run."""
    cmd = ["dvc", "repro", "-s", "--dry", "--ignore-errors"] + stages

    if glob:
        cmd.append("--glob")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        result.check_returncode()

        stages_to_run = []
        pattern = r"Running stage '([^']+)':"

        for line in result.stdout.splitlines():
            match = re.search(pattern, line)
            if match:
                stage_name = match.group(1)
                stages_to_run.append(stage_name)

        logger.info(f"Found {len(stages_to_run)} stages to commit")
        return stages_to_run

    except subprocess.CalledProcessError:
        logger.error(f"Error running fetching stages to commit: {result.stderr}")
        sys.exit(1)


def commit_stage(stage_name: str) -> bool:
    """Run dvc commit -f for a specific stage."""
    try:
        result = subprocess.run(
            ["dvc", "commit", "-f", stage_name],
            capture_output=True,
            text=True,
        )
        result.check_returncode()
        logger.info(f"Successfully committed stage {stage_name}")
        return True

    except subprocess.CalledProcessError:
        logger.error(f"Error committing stage {stage_name}: {result.stderr}")
        return False


if __name__ == "__main__":
    typer.run(main)
