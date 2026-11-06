"""Registro de prompts versionados.

Princípios (os mesmos de qualquer item de configuração):
  1. Identificação: todo prompt tem nome + versão semântica (nome@1.2.0).
  2. Imutabilidade: uma versão publicada nunca muda; alterar o texto exige nova versão.
  3. Rastreabilidade: prompts.lock.json guarda o SHA-256 de cada versão, então
     qualquer edição "silenciosa" é detectada (como um package-lock / dvc.lock).

Uso via linha de comando:
    python -m llmops.registry list
    python -m llmops.registry lock      # (re)gera prompts/prompts.lock.json
    python -m llmops.registry verify    # falha se uma versão publicada foi alterada
    python -m llmops.registry diff sentiment_classifier 1.0.0 1.1.0
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "prompts"
LOCK_FILE = "prompts.lock.json"


def parse_semver(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"versão inválida (esperado MAJOR.MINOR.PATCH): {version!r}")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def parse_ref(ref: str) -> tuple[str, str]:
    """'nome@1.0.0' -> ('nome', '1.0.0'); 'nome' -> ('nome', 'latest')."""
    name, _, version = ref.partition("@")
    return name, version or "latest"


def content_hash(path: Path) -> str:
    # Normaliza fim de linha para que Windows (CRLF) e Linux (LF) gerem o mesmo hash.
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    template: str
    input_variables: tuple[str, ...]
    sha256: str
    model_hints: dict = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    def render(self, **variables: str) -> str:
        placeholders = {f for _, f, _, _ in string.Formatter().parse(self.template) if f}
        missing = set(self.input_variables) - variables.keys()
        unexpected = variables.keys() - set(self.input_variables)
        if missing or unexpected:
            raise ValueError(f"{self.ref}: variáveis faltando={sorted(missing)} "
                             f"inesperadas={sorted(unexpected)}")
        if placeholders != set(self.input_variables):
            raise ValueError(f"{self.ref}: template usa {sorted(placeholders)}, "
                             f"mas declara {sorted(self.input_variables)}")
        return self.template.format(**variables)


class PromptRegistry:
    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root)

    def names(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def versions(self, name: str) -> list[str]:
        files = (self.root / name).glob("*.yaml")
        return sorted((f.stem for f in files), key=parse_semver)

    def get(self, ref: str, version: str | None = None) -> Prompt:
        name, ref_version = parse_ref(ref)
        version = version or ref_version
        available = self.versions(name)
        if not available:
            raise KeyError(f"prompt desconhecido: {name}")
        if version == "latest":
            version = available[-1]
        path = self.root / name / f"{version}.yaml"
        if not path.exists():
            raise KeyError(f"{name}@{version} não existe; disponíveis: {available}")

        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        if doc["name"] != name or doc["version"] != version:
            raise ValueError(f"{path}: name/version internos não batem com o caminho")
        return Prompt(
            name=name,
            version=version,
            template=doc["template"],
            input_variables=tuple(doc.get("input_variables", [])),
            sha256=content_hash(path),
            model_hints=doc.get("model_hints", {}),
        )

    # --- lock file -------------------------------------------------------
    def compute_lock(self) -> dict[str, dict[str, str]]:
        return {
            name: {v: content_hash(self.root / name / f"{v}.yaml") for v in self.versions(name)}
            for name in self.names()
        }

    def read_lock(self) -> dict[str, dict[str, str]]:
        path = self.root / LOCK_FILE
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def write_lock(self) -> None:
        lock = self.read_lock()
        for name, versions in self.compute_lock().items():
            for version, digest in versions.items():
                published = lock.get(name, {}).get(version)
                if published and published != digest:
                    raise RuntimeError(
                        f"{name}@{version} já publicado com outro conteúdo; crie uma nova versão")
                lock.setdefault(name, {})[version] = digest
        (self.root / LOCK_FILE).write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8", newline="\n")

    def verify_lock(self) -> list[str]:
        lock, current = self.read_lock(), self.compute_lock()
        errors = []
        for name, versions in current.items():
            for version, digest in versions.items():
                published = lock.get(name, {}).get(version)
                if published is None:
                    errors.append(f"{name}@{version}: versão nova fora do lock "
                                  f"(rode `python -m llmops.registry lock`)")
                elif published != digest:
                    errors.append(f"{name}@{version}: conteúdo alterado após publicação "
                                  f"(versões são imutáveis; crie uma nova versão)")
        for name, versions in lock.items():
            for version in versions.keys() - current.get(name, {}).keys():
                errors.append(f"{name}@{version}: versão publicada foi removida")
        return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Registro de prompts versionados")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("lock")
    sub.add_parser("verify")
    diff = sub.add_parser("diff")
    diff.add_argument("name")
    diff.add_argument("old")
    diff.add_argument("new")
    args = parser.parse_args(argv)

    reg = PromptRegistry(args.root)
    if args.cmd == "list":
        for name in reg.names():
            print(f"{name}: {', '.join(reg.versions(name))}")
    elif args.cmd == "lock":
        reg.write_lock()
        print(f"{LOCK_FILE} atualizado")
    elif args.cmd == "verify":
        errors = reg.verify_lock()
        for e in errors:
            print(f"ERRO {e}")
        if errors:
            return 1
        print("OK: todos os prompts conferem com o lock")
    elif args.cmd == "diff":
        old, new = reg.get(args.name, args.old), reg.get(args.name, args.new)
        sys.stdout.writelines(difflib.unified_diff(
            old.template.splitlines(keepends=True), new.template.splitlines(keepends=True),
            fromfile=old.ref, tofile=new.ref))
    return 0


if __name__ == "__main__":
    sys.exit(main())
