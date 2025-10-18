import json
from pathlib import Path

import typer
from hallucinations.metrics.rouge import compute_rouge_score
from hallucinations.metrics.squad import compute_squad_metrics

from hallucinations.utils.misc import save_json


def main(
    answers_file: Path = typer.Option(
        ...,
        help="Path to the answers JSON file (list of dicts {'answer': str, 'gold': str})",
    ),
) -> None:
    with answers_file.open("r") as file:
        answers = json.load(file)
    preds = [answer["prediction"] for answer in answers]
    golds = [answer["gold"] for answer in answers]

    squad_results = compute_squad_metrics(preds, golds, return_reduced=True, return_all=True)
    rouge_results = compute_rouge_score(preds, golds, return_reduced=True, return_all=True)

    results = {
        "reduced": {**squad_results["reduced"], **rouge_results["reduced"]},  # type: ignore[dict-item]
        "all": [sq | rg for sq, rg in zip(squad_results["all"], rouge_results["all"])],  # type: ignore
    }

    save_json(answers_file.with_name("metrics.json"), results)


if __name__ == "__main__":
    typer.run(main)
