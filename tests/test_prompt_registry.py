import shutil

import pytest

from llmops.registry import DEFAULT_ROOT, PromptRegistry, parse_semver


@pytest.fixture
def registry():
    return PromptRegistry()


@pytest.fixture
def tmp_registry(tmp_path):
    shutil.copytree(DEFAULT_ROOT, tmp_path / "prompts")
    return PromptRegistry(tmp_path / "prompts")


def test_semver_order_is_numeric_not_alphabetic():
    assert sorted(["1.10.0", "1.9.0", "1.2.3"], key=parse_semver) == ["1.2.3", "1.9.0", "1.10.0"]


def test_latest_resolves_to_highest_version(registry):
    assert registry.get("sentiment_classifier").version == "1.1.0"
    assert registry.get("sentiment_classifier@1.0.0").version == "1.0.0"


def test_render_rejects_missing_or_unknown_variables(registry):
    prompt = registry.get("sentiment_classifier@1.1.0")
    assert "Ótimo" in prompt.render(review="Ótimo")
    with pytest.raises(ValueError):
        prompt.render()
    with pytest.raises(ValueError):
        prompt.render(review="x", extra="y")


def test_repository_lock_is_consistent(registry):
    assert registry.verify_lock() == []


def test_editing_a_published_version_is_detected(tmp_registry):
    path = tmp_registry.root / "sentiment_classifier" / "1.0.0.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("Classifique", "Classifica"),
                    encoding="utf-8")
    errors = tmp_registry.verify_lock()
    assert any("sentiment_classifier@1.0.0" in e and "imutáveis" in e for e in errors)
    with pytest.raises(RuntimeError):
        tmp_registry.write_lock()


def test_new_version_must_be_locked(tmp_registry):
    folder = tmp_registry.root / "sentiment_classifier"
    text = (folder / "1.1.0.yaml").read_text(encoding="utf-8").replace("1.1.0", "1.2.0")
    (folder / "1.2.0.yaml").write_text(text, encoding="utf-8")
    assert any("1.2.0" in e for e in tmp_registry.verify_lock())
    tmp_registry.write_lock()
    assert tmp_registry.verify_lock() == []
