"""Validação e fingerprint de especificações de agentes (agents/*.yaml).

O comportamento de um agente depende de MUITO mais que o código: versão do
modelo, system prompt, ferramentas disponíveis, temperatura, guardrails.
Este módulo trata tudo isso como um único item de configuração:

  * valida o spec (versão fixada, prompt resolvível, ferramentas coerentes);
  * calcula um fingerprint = hash(spec + conteúdo do prompt referenciado).
    Se qualquer peça mudar, o fingerprint muda; é o "build id" do agente,
    útil para logs, auditoria e para saber qual configuração respondeu a um usuário.

    python -m llmops.agent_spec agents/support_agent.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

from llmops.registry import PromptRegistry, parse_ref, parse_semver


def load_spec(path: Path | str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def validate(spec: dict, registry: PromptRegistry) -> list[str]:
    errors: list[str] = []
    meta, body = spec.get("metadata", {}), spec.get("spec", {})

    try:
        parse_semver(str(meta.get("version", "")))
    except ValueError as exc:
        errors.append(f"metadata.version: {exc}")

    model = body.get("model", {})
    model_id = str(model.get("id", ""))
    if not model_id or "latest" in model_id:
        errors.append("spec.model.id deve fixar uma versão exata do modelo (sem 'latest')")
    if not 0 <= model.get("temperature", 0) <= 1:
        errors.append("spec.model.temperature deve estar entre 0 e 1")

    prompt_ref = body.get("system_prompt", "")
    _, version = parse_ref(prompt_ref)
    if version == "latest":
        errors.append(f"spec.system_prompt '{prompt_ref}' deve fixar a versão (nome@X.Y.Z)")
    else:
        try:
            registry.get(prompt_ref)
        except (KeyError, ValueError) as exc:
            errors.append(f"spec.system_prompt: {exc}")

    tool_names = [t.get("name") for t in body.get("tools", [])]
    if len(tool_names) != len(set(tool_names)):
        errors.append("spec.tools: nomes de ferramentas duplicados")
    for tool in body.get("tools", []):
        if "input_schema" not in tool:
            errors.append(f"spec.tools.{tool.get('name')}: falta input_schema")

    for name in body.get("guardrails", {}).get("require_human_approval", []):
        if name not in tool_names:
            errors.append(f"guardrails.require_human_approval: ferramenta inexistente '{name}'")

    return errors


def fingerprint(spec: dict, registry: PromptRegistry) -> str:
    prompt = registry.get(spec["spec"]["system_prompt"])
    payload = json.dumps({"spec": spec, "system_prompt_sha256": prompt.sha256}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Valida specs de agentes")
    parser.add_argument("specs", nargs="+")
    args = parser.parse_args(argv)

    registry = PromptRegistry()
    failed = False
    for path in args.specs:
        spec = load_spec(path)
        errors = validate(spec, registry)
        meta = spec.get("metadata", {})
        label = f"{meta.get('name')}@{meta.get('version')}"
        if errors:
            failed = True
            print(f"ERRO {path} ({label})")
            for e in errors:
                print(f"  - {e}")
        else:
            print(f"OK   {path} ({label}) fingerprint={fingerprint(spec, registry)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
