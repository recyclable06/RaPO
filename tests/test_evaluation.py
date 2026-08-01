import pytest

from rapo.evaluation import (
    aggregate_prediction_records,
    classification_answer_is_correct,
    compute_continual_metrics,
    extract_answer,
    normalize_class_name,
    pad_image_to_minimum_size,
    resolve_evaluator_settings,
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


def test_tiny_images_are_padded_to_the_model_spatial_factor():
    image_module = pytest.importorskip("PIL.Image")
    tiny_image = image_module.new("RGB", (27, 30), color="white")

    padded = pad_image_to_minimum_size(tiny_image, 28)

    assert padded.size == (28, 30)
    assert pad_image_to_minimum_size(padded, 28) is padded
    with pytest.raises(ValueError, match="minimum_size must be positive"):
        pad_image_to_minimum_size(tiny_image, 0)


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


def _strict_manifest():
    return {
        "schema_version": 1,
        "classes": [
            {
                "task_index": 1,
                "label": "cat",
                "test_images": ["task1/cat-a.jpg", "task1/cat-b.jpg"],
            },
            {
                "task_index": 2,
                "label": "dog",
                "test_images": ["task2/dog-a.jpg"],
            },
        ],
        "tasks": [{"task_index": 1}, {"task_index": 2}],
    }


def _strict_records():
    rows = []
    for after_task in (1, 2):
        rows.extend(
            {
                "after_task": after_task,
                "eval_task": 1,
                "relative_path": path,
                "completion": "<answer>cat</answer>",
                "target": "<answer>cat</answer>",
            }
            for path in ("task1/cat-a.jpg", "task1/cat-b.jpg")
        )
    rows.append(
        {
            "after_task": 2,
            "eval_task": 2,
            "relative_path": "task2/dog-a.jpg",
            "completion": "<answer>cat</answer>",
            "target": "<answer>dog</answer>",
        }
    )
    return rows


def test_prediction_aggregation_requires_exact_manifest_key_sets():
    assert aggregate_prediction_records(
        _strict_records(), data_manifest=_strict_manifest()
    ) == [
        {"after_task": 1, "eval_task": 1, "correct": 2, "total": 2},
        {"after_task": 2, "eval_task": 1, "correct": 2, "total": 2},
        {"after_task": 2, "eval_task": 2, "correct": 0, "total": 1},
    ]


@pytest.mark.parametrize(
    ("pollution", "message"),
    [
        ("duplicate", "Duplicate prediction key"),
        ("missing", "Missing prediction keys"),
        ("unknown", "Unknown prediction key"),
        ("target", "target does not match"),
    ],
)
def test_prediction_aggregation_rejects_four_pollution_classes(pollution, message):
    records = _strict_records()
    if pollution == "duplicate":
        records.append(dict(records[0]))
    elif pollution == "missing":
        records.pop(0)
    elif pollution == "unknown":
        records[0]["relative_path"] = "foreign.jpg"
    else:
        records[0]["target"] = "<answer>dog</answer>"

    with pytest.raises(ValueError, match=message):
        aggregate_prediction_records(records, data_manifest=_strict_manifest())


def test_evaluator_profile_is_explicit_and_formal_cannot_fallback():
    formal = resolve_evaluator_settings(
        profile_path="configs/formal_profile.json",
        torch_dtype=None,
        attention=None,
    )

    assert formal["torch_dtype"] == "bfloat16"
    assert formal["attention"] == "flash_attention_2"
    assert formal["profile_kind"] == "formal"
    with pytest.raises(ValueError, match="dtype must match"):
        resolve_evaluator_settings(
            profile_path="configs/formal_profile.json",
            torch_dtype="float16",
            attention=None,
        )

    legacy = resolve_evaluator_settings(
        profile_path=None,
        torch_dtype=None,
        attention=None,
    )
    assert (legacy["torch_dtype"], legacy["attention"]) == ("float16", "sdpa")


def test_prediction_lineage_must_match_every_row():
    records = _strict_records()
    expected = {"run_id": "run-a", "profile_sha256": "a" * 64}
    for record in records:
        record["lineage"] = dict(expected)
    records[-1]["lineage"]["run_id"] = "foreign"

    with pytest.raises(ValueError, match="lineage does not match"):
        aggregate_prediction_records(
            records,
            data_manifest=_strict_manifest(),
            expected_lineage=expected,
        )
