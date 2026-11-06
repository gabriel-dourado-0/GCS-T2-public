import copy
from pathlib import Path

import pytest

from llmops.agent_spec import fingerprint, load_spec, validate
from llmops.registry import PromptRegistry

SPEC_PATH = Path(__file__).resolve().parent.parent / "agents" / "support_agent.yaml"


@pytest.fixture
def registry():
    return PromptRegistry()


@pytest.fixture
def spec():
    return load_spec(SPEC_PATH)


def test_repository_spec_is_valid(spec, registry):
    assert validate(spec, registry) == []


def test_model_must_be_pinned(spec, registry):
    spec["spec"]["model"]["id"] = "claude-latest"
    assert any("model.id" in e for e in validate(spec, registry))


def test_prompt_must_be_pinned_and_exist(spec, registry):
    spec["spec"]["system_prompt"] = "support_agent_system"
    assert any("fixar a versão" in e for e in validate(spec, registry))
    spec["spec"]["system_prompt"] = "support_agent_system@9.9.9"
    assert any("não existe" in e for e in validate(spec, registry))


def test_guardrail_must_reference_existing_tool(spec, registry):
    spec["spec"]["guardrails"]["require_human_approval"] = ["apagar_banco"]
    assert any("apagar_banco" in e for e in validate(spec, registry))


def test_fingerprint_changes_with_any_configuration_change(spec, registry):
    original = fingerprint(spec, registry)
    assert fingerprint(copy.deepcopy(spec), registry) == original

    hotter = copy.deepcopy(spec)
    hotter["spec"]["model"]["temperature"] = 0.9
    assert fingerprint(hotter, registry) != original
