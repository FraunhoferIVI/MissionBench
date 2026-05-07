def task_gt_evaluation_prompt(
    task_gt_expected: str,
    task_gt_pred: str,
) -> str:
    """Build task_gt evaluation prompt."""
    return (
        "You are an evaluator. Compare EXPECTED task_gt with MODEL task_gt."
        " Determine if they match in meaning, allowing for minor variations"
        " in wording."
        " If the MODEL task_gt conveys the same intent and meaning as the"
        " EXPECTED task_gt,"
        " even if the wording is different, consider it a match."
        " If they differ significantly in intent or details, consider it"
        " not a match."
        " For letters or numbers, a 90 percentage similarity threshold"
        " can be used"
        " to determine a match; if one or two numbers/letters differ,"
        " it can still be considered success."
        " Respond with exactly one word: success or failed.\n\n"
        f"EXPECTED task_gt:\n{task_gt_expected}\n\n"
        f"MODEL task_gt:\n{task_gt_pred}\n"
    )
