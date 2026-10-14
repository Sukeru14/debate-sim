"""
Cliente de LLM. Abstrai a chamada ao modelo por trás de uma interface
simples, para que trocar de Llama 3.1 -> Qwen2.5 -> API externa
exija mudar só este arquivo (configuração), não o resto do sistema.

Usa a API REST nativa do Ollama (http://localhost:11434), sem dependências
externas além de 'requests'.

SOBRE FALHAS DE FORMATO JSON (relevante para modelos ~8B):
O modo 'format: json' do Ollama garante JSON sintaticamente válido, mas
NÃO garante que os campos venham com os nomes exatos pedidos no prompt --
isso é comportamento conhecido em modelos pequenos, especialmente quando
o contexto (histórico do debate) já está grande. Por isso:
  1. Toda falha de validação aqui LOGA a resposta bruta (stderr), para
     você conseguir inspecionar o que o modelo respondeu de fato.
  2. Há uma normalização de nomes de campo comuns (ex: "voto" -> "intencao_voto")
     antes de desistir, porque na prática o modelo costuma variar o nome
     do campo mais do que omitir a informação.
"""

import json
import sys
import time
import requests
from dataclasses import dataclass


@dataclass
class ConfigLLM:
    modelo: str = "llama3.1:8b"
    host: str = "http://localhost:11434"
    temperatura: float = 0.7
    seed: int | None = None       # fixar para reprodutibilidade (RQ4)
    max_tentativas_json: int = 5  # retries se o JSON vier malformado/incompleto
    num_ctx: int | None = None    # tamanho da janela de contexto (ver nota abaixo)


# Nomes alternativos que o modelo às vezes usa em vez do nome exato pedido.
# Mapeia variação -> nome canônico esperado pelo sistema.
ALIASES_CAMPOS = {
    "intencao_voto": ["voto", "intencao", "candidato_escolhido", "escolha"],
    "confianca": ["nivel_confianca", "certeza", "grau_confianca", "confidence"],
    "justificativa": ["razao", "motivo", "explicacao", "justificacao", "reasoning"],
}


def _normalizar_campos(data: dict, campos_esperados: list[str]) -> dict:
    """Tenta mapear nomes de campo alternativos para os nomes canônicos
    esperados, sem perder os dados caso o modelo tenha usado um sinônimo."""
    normalizado = dict(data)
    for campo_canonico in campos_esperados:
        if campo_canonico in normalizado:
            continue
        for alias in ALIASES_CAMPOS.get(campo_canonico, []):
            if alias in data:
                normalizado[campo_canonico] = data[alias]
                break
    return normalizado


class LLMClient:
    def __init__(self, config: ConfigLLM):
        self.config = config

    def _chamar(self, system: str, prompt: str, formato_json: bool = False) -> str:
        """Chamada bruta ao Ollama. Levanta exceção se a requisição falhar."""
        payload = {
            "model": self.config.modelo,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperatura,
            },
        }
        if self.config.seed is not None:
            payload["options"]["seed"] = self.config.seed
        if self.config.num_ctx is not None:
            payload["options"]["num_ctx"] = self.config.num_ctx
        if formato_json:
            payload["format"] = "json"  # Ollama força saída JSON sintaticamente válida

        resp = requests.post(f"{self.config.host}/api/generate", json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()["response"]

    def gerar_texto(self, system: str, prompt: str) -> str:
        """Para falas de candidato / narrativas de persona: texto livre."""
        return self._chamar(system, prompt, formato_json=False).strip()

    def gerar_json_validado(self, system: str, prompt: str, campos_obrigatorios: list[str]) -> dict:
        """
        Gera JSON e valida que os campos esperados existem, com:
          1. normalização de nomes de campo alternativos (ALIASES_CAMPOS)
          2. log da resposta bruta em CADA tentativa falha (stderr), para
             diagnóstico -- sem isso, falhas de formato são uma caixa-preta
          3. retry com lembrete explícito do formato esperado, reforçado
             a cada nova tentativa (ajuda quando o motivo foi o modelo
             "perder" a instrução em contexto longo)
        """
        ultimo_bruto = None
        ultimo_erro = None
        prompt_atual = prompt

        for tentativa in range(1, self.config.max_tentativas_json + 1):
            bruto = self._chamar(system, prompt_atual, formato_json=True)
            ultimo_bruto = bruto

            try:
                data = json.loads(bruto)
            except json.JSONDecodeError as e:
                ultimo_erro = f"JSON sintaticamente inválido: {e}"
                print(f"[LLMClient] tentativa {tentativa}/{self.config.max_tentativas_json} "
                      f"falhou ({ultimo_erro}). Resposta bruta: {bruto!r}", file=sys.stderr)
                time.sleep(0.5)
                prompt_atual = self._prompt_com_lembrete(prompt, campos_obrigatorios)
                continue

            data_normalizado = _normalizar_campos(data, campos_obrigatorios)
            faltando = [c for c in campos_obrigatorios if c not in data_normalizado]

            if not faltando:
                return data_normalizado

            ultimo_erro = f"Campos faltando após normalização: {faltando}"
            print(f"[LLMClient] tentativa {tentativa}/{self.config.max_tentativas_json} "
                  f"falhou ({ultimo_erro}). JSON recebido: {data!r}", file=sys.stderr)
            time.sleep(0.5)
            prompt_atual = self._prompt_com_lembrete(prompt, campos_obrigatorios)

        raise ValueError(
            f"Falha ao obter JSON com todos os campos obrigatórios "
            f"após {self.config.max_tentativas_json} tentativas.\n"
            f"Último erro: {ultimo_erro}\n"
            f"Última resposta bruta do modelo: {ultimo_bruto!r}"
        )

    @staticmethod
    def _prompt_com_lembrete(prompt_original: str, campos_obrigatorios: list[str]) -> str:
        """Reforça a instrução de formato no final do prompt -- útil quando
        o contexto já está longo e o modelo 'esquece' a instrução original
        que veio só no system prompt."""
        campos_str = ", ".join(f'"{c}"' for c in campos_obrigatorios)
        return (
            f"{prompt_original}\n\n"
            f"LEMBRETE IMPORTANTE: responda APENAS com um JSON contendo "
            f"exatamente os campos {campos_str}. Não inclua nenhum texto "
            f"fora do JSON."
        )
