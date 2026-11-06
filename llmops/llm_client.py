"""Interface mínima de cliente LLM + um modelo FALSO determinístico.

O FakeSentimentLLM existe para que os testes e a CI rodem offline, sem custo e
com resultado reproduzível. Ele imita um comportamento real importante: só
responde com rótulos que o PROMPT autoriza. Assim, mudar o prompt muda o
resultado da avaliação, que é exatamente o que queremos demonstrar.

Para usar um modelo real, implemente `complete()` chamando a API do provedor
(Anthropic, OpenAI, um modelo local etc.) e passe a instância para o avaliador.
"""
from __future__ import annotations

import re
from typing import Protocol

LABELS = ("positivo", "negativo", "neutro")
POSITIVE = {"excelente", "adorei", "ótima", "ótimo", "bom", "boa", "recomendo",
            "perfeitamente", "superou", "vale"}
NEGATIVE = {"péssimo", "péssima", "horrível", "ruim", "quebrou", "defeito",
            "arrependi", "rasgada", "não"}


class LLMClient(Protocol):
    def complete(self, prompt: str, **model_hints) -> str: ...


class FakeSentimentLLM:
    def complete(self, prompt: str, **model_hints) -> str:
        allowed = [label for label in LABELS if f'"{label}"' in prompt]
        review = re.findall(r"Avaliação:\s*(.*)", prompt)[-1].lower()
        words = set(re.findall(r"[\wáéíóúâêôãõç]+", review))
        pos, neg = len(words & POSITIVE), len(words & NEGATIVE)

        if pos > neg:
            guess = "positivo"
        elif neg > pos:
            guess = "negativo"
        else:
            guess = "neutro"
        if guess not in allowed:  # o prompt não oferece essa opção: o modelo "chuta"
            guess = "positivo" if pos >= neg else "negativo"
        return guess
