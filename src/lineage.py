"""Funções de linhagem: identificam exatamente QUAL código, QUAIS dados e QUAIS
parâmetros produziram um modelo. É o equivalente, para ML, de um build
reproduzível: dado o model_meta.json, é possível voltar ao estado exato que
gerou aqueles pesos.
"""
import hashlib
import json
import subprocess


def git_commit() -> str:
    """Commit atual do Git; sufixo '-dirty' se o CÓDIGO (src/) tiver alterações
    não commitadas. Dados e parâmetros já são identificados pelos seus hashes."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", "src"], text=True
        ).strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def file_md5(path: str) -> str:
    """MD5 do arquivo (mesmo algoritmo que o DVC usa nos arquivos .dvc/dvc.lock)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def params_hash(params: dict) -> str:
    """Hash canônico de um dicionário de parâmetros (independe da ordem das chaves)."""
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
