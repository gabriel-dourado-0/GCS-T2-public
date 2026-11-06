from llmops.evaluate_prompt import DEFAULT_GOLDEN, evaluate, load_cases
from llmops.llm_client import FakeSentimentLLM
from llmops.registry import PromptRegistry


def run(version: str) -> float:
    prompt = PromptRegistry().get("sentiment_classifier", version)
    return evaluate(prompt, load_cases(DEFAULT_GOLDEN), FakeSentimentLLM())["pass_rate"]


def test_latest_prompt_meets_quality_bar():
    assert run("latest") >= 0.9


def test_new_version_does_not_regress():
    assert run("1.1.0") > run("1.0.0")
