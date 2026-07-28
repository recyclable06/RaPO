import json

import pytest

from rapo.data import (
    build_imagenet_r_manifest,
    load_class_map,
    make_classification_prompt,
    visual_rft_rows,
)


def make_image_tree(tmp_path):
    image_root = tmp_path / "imagenet-r"
    class_map = {}
    for class_index in range(4):
        wnid = f"n{class_index:08d}"
        class_map[wnid] = f"class_{class_index}"
        class_directory = image_root / wnid
        class_directory.mkdir(parents=True)
        for image_index in range(3):
            (class_directory / f"{image_index}.jpg").write_bytes(b"image")
    return image_root, class_map


def build_small_manifest(image_root, class_map, **overrides):
    arguments = {
        "num_tasks": 2,
        "classes_per_task": 2,
        "shots_per_class": 1,
        "class_order_seed": 7,
        "sample_seed": 11,
    }
    arguments.update(overrides)
    return build_imagenet_r_manifest(image_root, class_map, **arguments)


def test_manifest_is_deterministic_disjoint_and_exhaustive(tmp_path):
    image_root, class_map = make_image_tree(tmp_path)

    first = build_small_manifest(image_root, class_map)
    second = build_small_manifest(image_root, class_map)

    assert first == second
    assert [task["train_size"] for task in first["tasks"]] == [2, 2]
    assert [task["test_size"] for task in first["tasks"]] == [4, 8]
    for class_record in first["classes"]:
        train = set(class_record["train_images"])
        test = set(class_record["test_images"])
        assert train.isdisjoint(test)
        assert len(train | test) == 3


def test_class_order_and_sample_seeds_are_independent(tmp_path):
    image_root, class_map = make_image_tree(tmp_path)

    baseline = build_small_manifest(image_root, class_map)
    changed_class_order = build_small_manifest(
        image_root, class_map, class_order_seed=8
    )
    changed_samples = build_small_manifest(image_root, class_map, sample_seed=12)

    assert [item["wnid"] for item in baseline["classes"]] != [
        item["wnid"] for item in changed_class_order["classes"]
    ]
    assert {
        item["wnid"]: item["train_images"] for item in baseline["classes"]
    } != {
        item["wnid"]: item["train_images"] for item in changed_samples["classes"]
    }


def test_visual_rft_rows_use_current_training_data_and_cumulative_vocabulary(
    tmp_path,
):
    image_root, class_map = make_image_tree(tmp_path)
    manifest = build_small_manifest(image_root, class_map)

    task_one_train = visual_rft_rows(manifest, image_root, task_index=1)
    task_two_train = visual_rft_rows(manifest, image_root, task_index=2)
    task_one_at_two = visual_rft_rows(
        manifest,
        image_root,
        task_index=2,
        eval_task=1,
    )

    assert len(task_one_train) == 2
    assert len(task_two_train) == 2
    assert len(task_one_at_two) == 4
    assert {row["task_index"] for row in task_two_train} == {2}
    assert all(
        class_name in task_two_train[0]["problem"]
        for class_name in manifest["tasks"][1]["seen_class_names"]
    )
    assert not all(
        class_name in task_one_train[0]["problem"]
        for class_name in manifest["tasks"][1]["seen_class_names"]
    )
    assert task_two_train[0]["solution"].startswith("<answer>")


def test_class_map_supports_standard_imagenet_index_layout(tmp_path):
    class_map_path = tmp_path / "imagenet_class_index.json"
    class_map_path.write_text(
        json.dumps(
            {
                "0": ["n01440764", "tench"],
                "1": ["n01443537", "goldfish"],
            }
        ),
        encoding="utf-8",
    )

    assert load_class_map(class_map_path) == {
        "n01440764": "tench",
        "n01443537": "goldfish",
    }


def test_class_map_supports_official_imagenet_r_readme(tmp_path):
    class_map_path = tmp_path / "README.txt"
    class_map_path.write_text(
        """ImageNet-R contains 30,000 images.

n01443537 goldfish
n01484850 great_white_shark
""",
        encoding="utf-8",
    )

    assert load_class_map(class_map_path) == {
        "n01443537": "goldfish",
        "n01484850": "great white shark",
    }


def test_manifest_rejects_class_count_mismatch(tmp_path):
    image_root, class_map = make_image_tree(tmp_path)

    with pytest.raises(ValueError, match="Expected 6 image classes"):
        build_imagenet_r_manifest(
            image_root,
            class_map,
            num_tasks=2,
            classes_per_task=3,
            shots_per_class=1,
        )


def test_prompt_matches_closed_set_contract():
    prompt = make_classification_prompt(["bell pepper", "candle"])

    assert "bell pepper, candle" in prompt
    assert "exactly one class name" in prompt
    assert "<think> </think>" in prompt
    assert "<answer> </answer>" in prompt
