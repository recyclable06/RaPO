import pytest

from rapo.evaluation import (
    aggregate_prediction_records,
    classification_answer_is_correct,
    compute_continual_metrics,
    extract_answer,
    normalize_class_name,
)


def test_classification_matching_follows_paper_normalization_and_exactness():
    assert normalize_class_name("  Bell_Pepper. ") == "bell pepper"
    assert classification_answer_is_correct(
        "<think>visual evidence</think><answer>bell-pepper</answer>",
        "<answer>Bell_Pepper.</answer>",
    )
    assert not classification_answer_is_correct(
        "<answer>catfish</answer>",
        "<answer>cat</answer>",
    )
    assert not classification_answer_is_correct("cat", "cat")
    assert extract_answer("<answer>a</answer><answer>b</answer>") is None


def test_prediction_aggregation_counts_each_accuracy_cell():
    records = [
        {
            "after_task": 1,
            "eval_task": 1,
            "completion": "<answer>cat</answer>",
            "target": "cat",
        },
        {
            "after_task": 1,
            "eval_task": 1,
            "completion": "<answer>dog</answer>",
            "target": "cat",
        },
    ]

    assert aggregate_prediction_records(records) == [
        {
            "after_task": 1,
            "eval_task": 1,
            "correct": 1,
            "total": 2,
        }
    ]


def test_continual_metrics_use_micro_last_accuracy_and_macro_forgetting():
    records = [
        {"after_task": 1, "eval_task": 1, "correct": 8, "total": 10},
        {"after_task": 2, "eval_task": 1, "correct": 7, "total": 10},
        {"after_task": 2, "eval_task": 2, "correct": 18, "total": 20},
        {"after_task": 3, "eval_task": 1, "correct": 6, "total": 10},
        {"after_task": 3, "eval_task": 2, "correct": 10, "total": 20},
        {"after_task": 3, "eval_task": 3, "correct": 9, "total": 10},
    ]

    metrics = compute_continual_metrics(records)

    assert metrics.last_accuracy == pytest.approx(25 / 40)
    assert metrics.forgetting == pytest.approx(0.3)
    assert metrics.accuracy_matrix == [
        [0.8, None, None],
        [0.7, 0.9, None],
        [0.6, 0.5, 0.9],
    ]


def test_continual_metrics_require_complete_lower_triangle():
    with pytest.raises(ValueError, match="Missing lower-triangular"):
        compute_continual_metrics(
            [
                {"after_task": 1, "eval_task": 1, "correct": 1, "total": 1},
                {"after_task": 2, "eval_task": 2, "correct": 1, "total": 1},
            ]
        )
