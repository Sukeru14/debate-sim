"""
Roda o experimento de "espectro político": 3 candidatos (direita,
esquerda, centro), cada um com uma plataforma e uma estratégia
argumentativa "adequada" já embutida na própria persona (não sorteada,
não alternada por fala) -- ver justificativa em CANDIDATOS_ESPECTRO.

Roda automaticamente as 3 combinações par-a-par possíveis entre os 3
candidatos (direita×esquerda, direita×centro, esquerda×centro), cada uma
como uma execução independente salva num banco separado, sempre
reaproveitando o MESMO lote de eleitores (--personas) para isolar a
variável "com quem o candidato debateu".

Uso:
    python rodar_embate_espectro.py --personas personas/lote_01.json --rodadas 5

Gera 3 arquivos SQLite em --db-dir (padrão "storage/"):
    embate_direita_vs_esquerda.db
    embate_direita_vs_centro.db
    embate_esquerda_vs_centro.db

Rode os gráficos/análise separadamente em cada um (analysis.gerar_graficos
--db storage/embate_direita_vs_esquerda.db --out graficos/direita_esquerda/,
etc.) já que são execuções logicamente distintas.
"""

import argparse
import itertools

from llm.client import LLMClient, ConfigLLM
from schemas import PersonaCandidato, EstrategiaArgumentativa, carregar_eleitores_json
from debate.orquestrador import OrquestradorDebate
from storage.db import Storage


# ---------------------------------------------------------------------------
# Os 3 candidatos do espectro político
# ---------------------------------------------------------------------------
#
# A estratégia argumentativa de cada um já vem embutida na persona (fixa
# para toda a execução, via `estrategia_a`/`estrategia_b` do
# OrquestradorDebate) -- não é sorteada nem alternada por fala.
#
# BASE DA ESCOLHA (declarar isso no TCC como pressuposto de modelagem,
# não como fato neutro sobre como a política "realmente" funciona):
# Graham, Haidt & Nosek (2009) documentam que a fundação moral
# "individualizing" (cuidado/dano, justiça/reciprocidade) tende a ser
# relativamente mais saliente à esquerda do espectro político, enquanto
# as fundações "binding" (lealdade, autoridade, pureza) tendem a receber
# peso mais equilibrado -- e frequentemente mais enfatizado -- à direita.
# Achado replicado em múltiplos países.
#
# Por isso direita e esquerda usam a MESMA categoria de estratégia
# (APELO_A_VALORES_MORAIS) -- a diferença está em QUAIS valores morais
# cada um invoca (texto do estilo_retorico), não na categoria retórica em
# si. Isso evita uma dicotomia sem base empírica (ex: "direita usa
# lógica, esquerda usa emoção") e reflete melhor o achado real: os dois
# lados usam apelo moral, só que a valores diferentes.
#
# O centro é modelado como deliberadamente menos moralizado e mais
# técnico/pragmático (DADOS_E_LOGICA) -- escolha razoável mas também uma
# simplificação; ajuste livremente o texto se não refletir o que você
# quer testar.

CANDIDATOS_ESPECTRO = {
    "direita": PersonaCandidato(
        id="direita",
        nome="Candidato Direita",
        plataforma=(
            "Defende redução do tamanho do Estado na economia, incentivo à "
            "livre iniciativa e ao empreendedorismo, políticas de segurança "
            "pública mais rígidas, valorização da família tradicional e das "
            "forças de segurança, e menor intervenção governamental na vida "
            "privada dos cidadãos."
        ),
        estilo_retorico=(
            "firme e assertivo; recorre a apelos de ordem, tradição, "
            "autoridade e proteção de valores compartilhados; enfatiza "
            "responsabilidade individual, mérito e disciplina."
        ),
    ),
    "esquerda": PersonaCandidato(
        id="esquerda",
        nome="Candidato Esquerda",
        plataforma=(
            "Defende ampliação de direitos sociais e trabalhistas, "
            "redistribuição de renda, políticas afirmativas para grupos "
            "historicamente marginalizados, maior regulação do mercado e "
            "investimento público em saúde e educação."
        ),
        estilo_retorico=(
            "engajado e solidário; recorre a apelos de cuidado com os mais "
            "vulneráveis, justiça social e igualdade; enfatiza "
            "responsabilidade coletiva e redução de desigualdades."
        ),
    ),
    "centro": PersonaCandidato(
        id="centro",
        nome="Candidato Centro",
        plataforma=(
            "Defende reformas graduais e negociadas, equilíbrio fiscal "
            "responsável combinado com proteção social básica, governança "
            "técnica e institucional, e busca de consensos entre diferentes "
            "setores da sociedade."
        ),
        estilo_retorico=(
            "ponderado e pragmático; recorre a dados, evidências técnicas e "
            "histórico de resultados concretos; evita apelos ideológicos "
            "fortes e se apresenta como alternativa moderada."
        ),
    ),
}

ESTRATEGIA_POR_POSICAO = {
    "direita": EstrategiaArgumentativa.APELO_A_VALORES_MORAIS,
    "esquerda": EstrategiaArgumentativa.APELO_A_VALORES_MORAIS,
    "centro": EstrategiaArgumentativa.DADOS_E_LOGICA,
}

TEMAS_PADRAO = [
    "economia e geração de empregos",
    "saúde pública",
    "segurança pública",
    "educação",
    "meio ambiente e sustentabilidade",
]


def rodar_confronto(
    nome_a: str, nome_b: str, eleitores, config_llm, n_rodadas: int,
    db_dir: str, max_rodadas_historico: int | None,
):
    candidato_a = CANDIDATOS_ESPECTRO[nome_a]
    candidato_b = CANDIDATOS_ESPECTRO[nome_b]
    estrategia_a = ESTRATEGIA_POR_POSICAO[nome_a]
    estrategia_b = ESTRATEGIA_POR_POSICAO[nome_b]

    db_path = f"{db_dir}/embate_{nome_a}_vs_{nome_b}.db"
    storage = Storage(db_path=db_path)

    orquestrador = OrquestradorDebate(
        candidato_a=candidato_a,
        candidato_b=candidato_b,
        eleitores=eleitores,
        temas=TEMAS_PADRAO,
        config_llm=config_llm,
        storage=storage,
        descricao_run=f"Embate de espectro: {nome_a} vs {nome_b}",
        max_workers_eleitores=4,
        alternar_abertura=True,
        max_rodadas_no_historico=max_rodadas_historico,
        estrategia_a=estrategia_a,
        estrategia_b=estrategia_b,
    )
    print(f"\n=== Confronto: {nome_a} vs {nome_b} -> {db_path} ===")
    run_id = orquestrador.rodar_simulacao(n_rodadas=n_rodadas)
    storage.close()
    return db_path, run_id


def main():
    parser = argparse.ArgumentParser(
        description="Roda os 3 confrontos par-a-par do espectro político (direita/esquerda/centro)"
    )
    parser.add_argument("--personas", type=str, required=True,
                         help="JSON de eleitores (gerado por gerar_personas.py) -- "
                              "o MESMO lote é reaproveitado nos 3 confrontos")
    parser.add_argument("--rodadas", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modelo", type=str, default="llama3.1")
    parser.add_argument("--db-dir", type=str, default="storage",
                         help="diretório onde salvar os 3 bancos SQLite (um por confronto)")
    parser.add_argument("--num-ctx", type=int, default=None)
    parser.add_argument("--max-rodadas-historico", type=int, default=3)
    args = parser.parse_args()

    print(f"Carregando eleitores de {args.personas}...")
    eleitores = carregar_eleitores_json(args.personas)
    print(f"  {len(eleitores)} eleitores carregados (mesmo lote nos 3 confrontos).")

    config_llm = ConfigLLM(modelo=args.modelo, temperatura=0.7, seed=args.seed, num_ctx=args.num_ctx)
    max_rodadas_historico = args.max_rodadas_historico if args.max_rodadas_historico else None

    import os
    os.makedirs(args.db_dir, exist_ok=True)

    resultados = []
    for nome_a, nome_b in itertools.combinations(CANDIDATOS_ESPECTRO.keys(), 2):
        db_path, run_id = rodar_confronto(
            nome_a, nome_b, eleitores, config_llm, args.rodadas, args.db_dir, max_rodadas_historico,
        )
        resultados.append((nome_a, nome_b, db_path, run_id))

    print("\n=== Resumo dos 3 confrontos ===")
    for nome_a, nome_b, db_path, run_id in resultados:
        print(f"  {nome_a} vs {nome_b}: run_id={run_id}, banco={db_path}")


if __name__ == "__main__":
    main()
