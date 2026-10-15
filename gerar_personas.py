"""
ETAPA 1: gera o lote de eleitores e salva em JSON (legível e editável à
mão). O mesmo lote é reaproveitado em todos os confrontos de main.py.

    # perfil do arquivo bruto da PNAD Contínua (FTP do IBGE)
    python gerar_personas.py --n 30 --arquivo dados/PNADC_012026.txt --out personas/lote_01.json

    # perfil de um CSV extraído via Base dos Dados (extrair_microdados.py)
    python gerar_personas.py --n 30 --fonte-perfil csv --arquivo dados/pnadc_recorte.csv --out personas/lote_01.json

--fonte-tracos correlacionado usa a estrutura de correlação da literatura
para Big Five e Fundamentos Morais (ver persona/gerador_subjetivo_correlacionado.py).
"""

import argparse

from llm.client import ConfigLLM, LLMClient
from persona.gerador_subjetivo import GeradorTracosIndependente, gerar_persona
from schemas import salvar_eleitores_json


def criar_gerador_perfil(fonte: str, arquivo: str, seed: int):
    if fonte == "pnadc_bruto":
        from persona.gerador_estatistico_pnadc_bruto import GeradorPerfilPNADCBruto
        return GeradorPerfilPNADCBruto(arquivo, seed=seed)
    from persona.gerador_estatistico_real import GeradorPerfilMicrodadosCSV
    return GeradorPerfilMicrodadosCSV(arquivo, seed=seed)


def criar_gerador_tracos(fonte: str, seed: int):
    if fonte == "independente":
        return GeradorTracosIndependente(seed=seed)
    from persona.gerador_subjetivo_correlacionado import GeradorTracosCorrelacionado
    return GeradorTracosCorrelacionado(seed=seed)


def main():
    parser = argparse.ArgumentParser(description="Gera um lote de personas de eleitores")
    parser.add_argument("--n", type=int, default=10, help="número de eleitores")
    parser.add_argument("--arquivo", required=True, help="microdados da PNAD (.txt bruto ou .csv/.parquet)")
    parser.add_argument("--fonte-perfil", choices=["pnadc_bruto", "csv"], default="pnadc_bruto")
    parser.add_argument("--fonte-tracos", choices=["independente", "correlacionado"], default="independente")
    parser.add_argument("--out", default="personas/lote_eleitores.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modelo", default="llama3.1:8b")
    args = parser.parse_args()

    gerador_perfil = criar_gerador_perfil(args.fonte_perfil, args.arquivo, args.seed)
    gerador_tracos = criar_gerador_tracos(args.fonte_tracos, args.seed)
    llm = LLMClient(ConfigLLM(modelo=args.modelo, temperatura=0.8, seed=args.seed))

    eleitores = []
    for i, perfil in enumerate(gerador_perfil.amostrar_lote(args.n), start=1):
        persona = gerar_persona(f"eleitor_{i:03d}", perfil, gerador_tracos, llm)
        eleitores.append(persona)
        print(f"  [{i}/{args.n}] {persona.id}: {perfil.idade} anos, {perfil.genero}, {perfil.uf}")

    salvar_eleitores_json(eleitores, args.out)
    print(f"\n{len(eleitores)} eleitores salvos em {args.out}")


if __name__ == "__main__":
    main()
