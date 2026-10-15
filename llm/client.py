"""
Cliente do LLM (API REST nativa do Ollama). Trocar de modelo ou de
provedor deve exigir mudanças só neste arquivo.

Sobre JSON com modelos ~8B: `format: json` garante JSON sintaticamente
válido, mas não que os campos venham com os nomes pedidos. Por isso
`gerar_json` normaliza sinônimos comuns, valida o conteúdo e, se falhar,
tenta de novo reforçando o formato no prompt (e mudando a seed, senão a
nova tentativa repetiria a mesma saída).
"""

from dataclasses import dataclass
import json
import sys
from typing import Callable

import requests


@dataclass
class ConfigLLM:
    modelo: str = "llama3.1:8b"
    host: str = "http://localhost:11434"
    temperatura: float = 0.7
    seed: int | None = None
    # O padrão do Ollama (2048-4096) é pequeno para persona + histórico do
    # debate; quando estoura, o Ollama corta o INÍCIO do prompt sem avisar.
    num_ctx: int = 8192
    max_tentativas_json: int = 5
    timeout: int = 300


ALIASES_CAMPOS = {
    "intencao_voto": ["voto", "intencao", "candidato", "escolha"],
    "confianca": ["nivel_confianca", "certeza", "confidence"],
    "justificativa": ["razao", "motivo", "explicacao", "justificacao"],
}


class LLMClient:
    def __init__(self, config: ConfigLLM):
        self.config = config

    def _chamar(self, system: str, prompt: str, formato_json: bool = False, tentativa: int = 0) -> str:
        opcoes = {"temperature": self.config.temperatura, "num_ctx": self.config.num_ctx}
        if self.config.seed is not None:
            opcoes["seed"] = self.config.seed + tentativa
        payload = {
            "model": self.config.modelo,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": opcoes,
        }
        if formato_json:
            payload["format"] = "json"

        resp = requests.post(f"{self.config.host}/api/generate", json=payload, timeout=self.config.timeout)
        resp.raise_for_status()
        return resp.json()["response"].strip()

    def gerar_texto(self, system: str, prompt: str) -> str:
        return self._chamar(system, prompt)

    def gerar_json(
        self,
        system: str,
        prompt: str,
        campos: list[str],
        validar: Callable[[dict], dict] | None = None,
    ) -> dict:
        """
        Gera um JSON com `campos` obrigatórios. `validar` recebe o dict e
        devolve a versão normalizada, ou levanta ValueError para pedir
        nova tentativa.
        """
        prompt_atual = prompt
        erro = None
        for tentativa in range(self.config.max_tentativas_json):
            bruto = self._chamar(system, prompt_atual, formato_json=True, tentativa=tentativa)
            try:
                data = _normalizar_campos(json.loads(bruto), campos)
                faltando = [c for c in campos if c not in data]
                if faltando:
                    raise ValueError(f"campos faltando: {faltando}")
                return validar(data) if validar else data
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                erro = e
                print(f"[LLM] tentativa {tentativa + 1}/{self.config.max_tentativas_json} "
                      f"falhou ({e}). Resposta: {bruto!r}", file=sys.stderr)
                prompt_atual = (
                    f"{prompt}\n\nLEMBRETE: responda APENAS com um JSON contendo exatamente "
                    f"os campos {', '.join(campos)}, nos formatos pedidos."
                )
        raise ValueError(f"JSON inválido após {self.config.max_tentativas_json} tentativas: {erro}")


def _normalizar_campos(data: dict, campos: list[str]) -> dict:
    if not isinstance(data, dict):
        raise ValueError("a resposta não é um objeto JSON")
    normalizado = dict(data)
    for campo in campos:
        if campo in normalizado:
            continue
        for alias in ALIASES_CAMPOS.get(campo, []):
            if alias in data:
                normalizado[campo] = data[alias]
                break
    return normalizado
