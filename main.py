"""
ETAPA 2 do pipeline: carrega um lote de personas JÁ GERADO (ver
gerar_personas.py) e roda a simulação de debate.

Pipeline completo:
    1. python gerar_personas.py --n 10 --out personas/lote_01.json
    2. python main.py --personas personas/lote_01.json

Mantendo as etapas separadas você pode reusar o MESMO lote de eleitores
em várias rodadas de simulação (ex: testando diferentes temas ou
diferentes candidatos), controlando essa variável.
"""

import argparse

from llm.client import LLMClient, ConfigLLM
from schemas import PersonaCandidato, carregar_eleitores_json, EstrategiaArgumentativa
from debate.orquestrador import OrquestradorDebate
from storage.db import Storage


TEMAS_PADRAO = [
    "economia e geração de empregos",
    "saúde pública",
    "segurança pública",
    "educação",
    "meio ambiente e sustentabilidade",
]

ESTRATEGIAS_VALIDAS = [e.value for e in EstrategiaArgumentativa]


def main():
    parser = argparse.ArgumentParser(description="Roda a simulação de debate eleitoral")
    parser.add_argument("--personas", type=str, required=True,
                         help="caminho do JSON de personas (gerado por gerar_personas.py)")
    parser.add_argument("--rodadas", type=int, default=3, help="número de rodadas")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modelo", type=str, default="llama3.1")
    parser.add_argument("--db", type=str, default="storage/debate_sim.db")
    parser.add_argument("--fixar-abertura", action="store_true",
                         help="desativa a alternância de quem abre cada rodada -- candidato A "
                              "sempre abre. Por padrão (sem esta flag) a abertura ALTERNA a cada "
                              "rodada, para não introduzir uma assimetria sistemática que não "
                              "existe em debates reais (a Lei das Eleições determina ordem por "
                              "sorteio).")
    parser.add_argument("--max-rodadas-historico", type=int, default=3,
                         help="quantas rodadas anteriores incluir no contexto passado ao "
                              "modelo (None/0 = histórico completo). Reduz risco de falha "
                              "de formato JSON em debates longos com modelos pequenos.")
    parser.add_argument("--num-ctx", type=int, default=None,
                         help="tamanho da janela de contexto do Ollama (ex: 8192). "
                              "Se o modelo truncar o início do prompt silenciosamente, "
                              "aumentar este valor pode ajudar (padrão do Ollama é 2048, "
                              "geralmente pequeno demais para este caso de uso).")
    parser.add_argument("--estrategia-a", type=str, default=None, choices=ESTRATEGIAS_VALIDAS,
                         help="se definida, o candidato A usa SEMPRE esta estratégia durante "
                              "toda a execução (em vez de sortear a cada fala). Use junto com "
                              "--estrategia-b para montar 'embates' entre estratégias fixas "
                              "(RQ3) -- rode uma execução por combinação, reaproveitando o "
                              "mesmo --personas em todas.")
    parser.add_argument("--estrategia-b", type=str, default=None, choices=ESTRATEGIAS_VALIDAS,
                         help="mesma ideia de --estrategia-a, mas para o candidato B.")
    args = parser.parse_args()

    print(f"Carregando eleitores de {args.personas}...")
    eleitores = carregar_eleitores_json(args.personas)
    print(f"  {len(eleitores)} eleitores carregados.")

    candidato_a = PersonaCandidato(
        id="A",
        nome="Candidato A",
        plataforma=(
            "Defende aumento de investimento em programas sociais, "
            "regulação ambiental mais forte e ampliação de direitos trabalhistas."
        ),
        estilo_retorico="caloroso e empático, fala em histórias pessoais",
    )
    candidato_b = PersonaCandidato(
        id="B",
        nome="Candidato B",
        plataforma=(
            "Defende redução de impostos, desburocratização econômica e "
            "mais autonomia para estados e municípios."
        ),
        estilo_retorico="direto e técnico, fala em números e exemplos concretos",
    )

    config_llm = ConfigLLM(modelo=args.modelo, temperatura=0.7, seed=args.seed, num_ctx=args.num_ctx)
    storage = Storage(db_path=args.db)

    max_rodadas_historico = args.max_rodadas_historico if args.max_rodadas_historico else None
    estrategia_a = EstrategiaArgumentativa(args.estrategia_a) if args.estrategia_a else None
    estrategia_b = EstrategiaArgumentativa(args.estrategia_b) if args.estrategia_b else None

    descricao_run = f"Simulação com {len(eleitores)} eleitores, {args.rodadas} rodadas"
    if estrategia_a or estrategia_b:
        descricao_run += f" | estrategia_a={args.estrategia_a} estrategia_b={args.estrategia_b}"

    orquestrador = OrquestradorDebate(
        candidato_a=candidato_a,
        candidato_b=candidato_b,
        eleitores=eleitores,
        temas=TEMAS_PADRAO,
        config_llm=config_llm,
        storage=storage,
        descricao_run=descricao_run,
        max_workers_eleitores=4,
        alternar_abertura=not args.fixar_abertura,
        max_rodadas_no_historico=max_rodadas_historico,
        estrategia_a=estrategia_a,
        estrategia_b=estrategia_b,
    )

    run_id = orquestrador.rodar_simulacao(n_rodadas=args.rodadas)
    storage.close()

    print(f"\nConcluído. run_id = {run_id}")
    print(f"Dados salvos em {args.db}")


if __name__ == "__main__":
    main()
