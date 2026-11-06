"""Avaliação de regressão de prompts contra um golden set.

Prompts são código: uma alteração "inofensiva" no texto pode derrubar a
qualidade. Por isso cada versão é avaliada contra um conjunto fixo de casos
(evals/*.jsonl) e a CI bloqueia versões abaixo do limite.

    python -m llmops.evaluate_prompt sentiment_classifier@latest --min-pass-rate 0.9
    python -m llmops.evaluate_prompt sentiment_classifier --compare 1.0.0 1.1.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llmops.llm_client import FakeSentimentLLM, LLMClient
from llmops.registry import Prompt, PromptRegistry

DEFAULT_GOLDEN = Path(__file__).resolve().parent.parent / "evals" / "sentiment_golden.jsonl"


def load_cases(path: Path | str) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def evaluate(prompt: Prompt, cases: list[dict], client: LLMClient) -> dict:
    results = []
    for case in cases:
        output = client.complete(prompt.render(review=case["review"]), **prompt.model_hints)
        answer = output.strip().lower()
        results.append({**case, "output": answer, "passed": answer == case["expected"]})
    passed = sum(r["passed"] for r in results)
    return {"prompt": prompt.ref, "sha256": prompt.sha256[:12],
            "pass_rate": round(passed / len(results), 4), "results": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", help="nome ou nome@versão")
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--min-pass-rate", type=float, default=0.0)
    parser.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    registry, client, cases = PromptRegistry(), FakeSentimentLLM(), load_cases(args.golden)

    if args.compare:
        old, new = (evaluate(registry.get(args.prompt, v), cases, client) for v in args.compare)
        print(f"{old['prompt']}: {old['pass_rate']:.0%}  ->  {new['prompt']}: {new['pass_rate']:.0%}")
        for o, n in zip(old["results"], new["results"]):
            if o["passed"] != n["passed"]:
                mark = "CORRIGIDO" if n["passed"] else "REGRESSÃO"
                print(f"  [{mark}] {o['review']!r}: {o['output']} -> {n['output']} "
                      f"(esperado {o['expected']})")
        return 1 if new["pass_rate"] < old["pass_rate"] else 0

    report = evaluate(registry.get(args.prompt), cases, client)
    print(f"{report['prompt']} (sha256 {report['sha256']}): pass rate {report['pass_rate']:.0%}")
    for r in report["results"]:
        if args.verbose or not r["passed"]:
            print(f"  [{'OK' if r['passed'] else 'FALHOU'}] {r['review']!r} -> {r['output']} "
                  f"(esperado {r['expected']})")
    if report["pass_rate"] < args.min_pass_rate:
        print(f"FALHA: pass rate abaixo do mínimo {args.min_pass_rate:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
