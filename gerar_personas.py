"""
ETAPA 1 do pipeline: gera as personas dos eleitores e salva em JSON.

Roda separado da simulação de propósito -- assim você pode:
- inspecionar/editar as personas manualmente antes de rodar o debate
- reusar o MESMO lote de eleitores em várias simulações (ex: comparar
  estratégias de debate diferentes com os mesmos eleitores, controlando
  essa variável)
- gerar um lote grande uma vez só (a geração via LLM tem custo de tempo)
  e reaproveitar em múltiplos experimentos

Uso (perfil sociodemográfico simulado -- distribuições aproximadas):
    python gerar_personas.py --n 20 --out personas/lote_01.json --seed 42

Uso (perfil sociodemográfico de microdados REAIS da PNAD Contínua,
arquivo bruto baixado do FTP do IBGE):
    python gerar_personas.py --n 20 --out personas/lote_01.json \\
        --fonte-perfil pnadc_bruto --arquivo-pnadc dados/PNADC_012026.txt

Uso (traços de personalidade com correlação entre si, informada pela
literatura -- ver persona/gerador_subjetivo_correlacionado.py):
    python gerar_personas.py --n 20 --out personas/lote_01.json \\
        --fonte-tracos correlacionado

Com --fonte-perfil pnadc_bruto: idade, sexo, UF/região, raça/cor,
situação do domicílio e peso amostral vêm REAIS do arquivo da PNAD
Contínua (respeitando o peso amostral). Escolaridade e renda continuam
aproximadas -- ver docstring de persona/gerador_estatistico_pnadc_bruto.py
para o porquê. Religião foi RETIRADA do perfil (decisão do orientador,
volta quando houver dados reais confiáveis) -- todo perfil traz
religiao="não_modelada".

Com --fonte-tracos correlacionado: Big Five e Fundamentos Morais deixam
de ser amostrados de forma independente entre si e passam a respeitar a
estrutura de correlação documentada na literatura (metatraços Stability/
Plasticity no Big Five; clusters individualizing/binding nos Fundamentos
Morais) -- ver docstring de persona/gerador_subjetivo_correlacionado.py.
"""

import argparse

from llm.client import LLMClient, ConfigLLM
from persona.gerador_subjetivo import GeradorTracosSubjetivos, gerar_persona_completa
from schemas import salvar_eleitores_json


def _criar_gerador_tracos(fonte_tracos: str, seed: int, condicionar_religiosidade: bool):
    if fonte_tracos == "independente":
        return GeradorTracosSubjetivos(seed=seed)

    if fonte_tracos == "correlacionado":
        from persona.gerador_subjetivo_correlacionado import GeradorTracosSubjetivosCorrelacionado
        return GeradorTracosSubjetivosCorrelacionado(
            seed=seed, condicionar_religiosidade=condicionar_religiosidade,
        )

    raise ValueError(f"fonte_tracos desconhecida: {fonte_tracos!r}")


def _criar_gerador_perfil(fonte_perfil: str, seed: int, arquivo_pnadc: str | None):
    if fonte_perfil == "simulado":
        from persona.gerador_estatistico import GeradorPerfilSociodemografico
        return GeradorPerfilSociodemografico(seed=seed)

    if fonte_perfil == "pnadc_bruto":
        if not arquivo_pnadc:
            raise ValueError(
                "--arquivo-pnadc é obrigatório quando --fonte-perfil pnadc_bruto. "
                "Aponte para o .txt de microdados baixado do FTP do IBGE."
            )
        from persona.gerador_estatistico_pnadc_bruto import GeradorPerfilSociodemograficoPNADCBruto
        return GeradorPerfilSociodemograficoPNADCBruto(caminho_arquivo=arquivo_pnadc, seed=seed)

    raise ValueError(f"fonte_perfil desconhecida: {fonte_perfil!r}")


def gerar_lote_eleitores(
    n: int,
    seed: int,
    modelo_llm: str = "llama3.1",
    fonte_perfil: str = "simulado",
    arquivo_pnadc: str | None = None,
    fonte_tracos: str = "independente",
    condicionar_religiosidade: bool = False,
) -> list:
    config_llm = ConfigLLM(modelo=modelo_llm, temperatura=0.8, seed=seed)
    llm = LLMClient(config_llm)

    gerador_perfil = _criar_gerador_perfil(fonte_perfil, seed, arquivo_pnadc)
    gerador_tracos = _criar_gerador_tracos(fonte_tracos, seed, condicionar_religiosidade)

    perfis = gerador_perfil.amostrar_lote(n)
    eleitores = []
    for i, perfil in enumerate(perfis):
        persona = gerar_persona_completa(
            eleitor_id=f"eleitor_{i+1:03d}",
            perfil=perfil,
            llm=llm,
            gerador_tracos=gerador_tracos,
        )
        eleitores.append(persona)
        print(f"  [{i+1}/{n}] {persona.id}: {perfil.idade} anos, {perfil.regiao}/{perfil.uf}, "
              f"abertura={persona.big_five.abertura:.2f}, "
              f"cuidado_dano={persona.fundamentos_morais.cuidado_dano:.2f}")

    return eleitores


def main():
    parser = argparse.ArgumentParser(description="Gera lote de personas de eleitores")
    parser.add_argument("--n", type=int, default=10, help="número de eleitores a gerar")
    parser.add_argument("--out", type=str, default="personas/lote_eleitores.json",
                         help="caminho do arquivo JSON de saída")
    parser.add_argument("--seed", type=int, default=42, help="seed para reprodutibilidade")
    parser.add_argument("--modelo", type=str, default="llama3.1", help="modelo Ollama a usar")
    parser.add_argument("--fonte-perfil", type=str, default="simulado",
                         choices=["simulado", "pnadc_bruto"],
                         help="'simulado' usa distribuições aproximadas; 'pnadc_bruto' usa "
                              "o arquivo de microdados real da PNAD Contínua (ver --arquivo-pnadc)")
    parser.add_argument("--arquivo-pnadc", type=str, default=None,
                         help="caminho do .txt de microdados bruto da PNAD Contínua "
                              "(obrigatório se --fonte-perfil pnadc_bruto)")
    parser.add_argument("--fonte-tracos", type=str, default="independente",
                         choices=["independente", "correlacionado"],
                         help="'independente' amostra Big Five/Fundamentos Morais sem "
                              "correlação entre si (default histórico); 'correlacionado' "
                              "usa a estrutura de correlação informada pela literatura "
                              "(ver persona/gerador_subjetivo_correlacionado.py)")
    parser.add_argument("--condicionar-religiosidade", action="store_true",
                         help="(só com --fonte-tracos correlacionado) SEM EFEITO ATUALMENTE -- "
                              "religião foi retirada do perfil (ver README). Desloca levemente "
                              "a média dos fundamentos morais binding conforme a religião do "
                              "perfil quando essa variável voltar a ser modelada.")
    args = parser.parse_args()

    print(f"Gerando {args.n} eleitores (seed={args.seed}, modelo={args.modelo}, "
          f"fonte_perfil={args.fonte_perfil}, fonte_tracos={args.fonte_tracos})...")
    eleitores = gerar_lote_eleitores(
        n=args.n, seed=args.seed, modelo_llm=args.modelo,
        fonte_perfil=args.fonte_perfil, arquivo_pnadc=args.arquivo_pnadc,
        fonte_tracos=args.fonte_tracos, condicionar_religiosidade=args.condicionar_religiosidade,
    )

    salvar_eleitores_json(eleitores, args.out)
    print(f"\n{len(eleitores)} eleitores salvos em: {args.out}")


if __name__ == "__main__":
    main()
