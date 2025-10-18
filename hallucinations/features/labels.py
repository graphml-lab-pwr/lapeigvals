CORRECT_ANSWER = 0
INCORRECT_ANSWER = 1
INVALID_ANSWER = -1


def compute_labels(
    llm_judge_results: list[str],
    metric_results: list[dict[str, float]],
) -> tuple[list[int], list[bool]]:
    """Computes labels and returns valid label mask"""
    labels = []
    for judge, m_values in zip(llm_judge_results, metric_results):
        if m_values["squad_f1"] >= 0.99 or m_values["rougeL_fmeasure"] >= 0.99:
            labels.append(CORRECT_ANSWER)
        elif judge == "incorrect" and m_values["rougeL_fmeasure"] >= 0.75:
            labels.append(INVALID_ANSWER)
        elif judge == "correct":
            labels.append(CORRECT_ANSWER)
        elif judge == "incorrect":
            labels.append(INCORRECT_ANSWER)
        else:
            labels.append(INVALID_ANSWER)

    valid_labels_mask = [label != INVALID_ANSWER for label in labels]
    return labels, valid_labels_mask
