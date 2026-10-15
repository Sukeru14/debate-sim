"""
ETAPA 2: roda os confrontos entre os tipos de candidato, todos com o
MESMO lote de eleitores (gerado por gerar_personas.py).

    python main.py --personas personas/lote_01.json --rodadas 5

Por padrão roda os 3 confrontos possíveis (emocional x logico,
emocional x credibilidade, logico x credibilidade). Tudo vai para um
único banco SQLite, com uma linha em `runs` por confronto/repetição.

--contrabalancear roda também cada confronto com os rótulos invertidos
(quem era "Candidato 1" vira "Candidato 2"), para controlar viés de
posição/rótulo do modelo. Dobra o tempo de execução.
"""

import argparse
from itertools import combinations
import uuid

from debate.candidatos import TIPOS
from debate.orquestrador import OrquestradorDebate
from llm.client import ConfigLLM, LLMClient
from schemas import carregar_eleitores_json
from storage.db import Storage


def _parse_confronto(texto: str) -> tuple[str, str]:
    partes = texto.split(":")
    if len(partes) != 2 or not all(p in TIPOS for p in partes) or partes[0] == partes[1]:
        raise argparse.ArgumentTypeError(
            f"confronto inválido {texto!r}; use tipo1:tipo2 com tipos em {list(TIPOS)}"
        )
    return partes[0], partes[1]


def main():
    parser = argparse.ArgumentParser(description="Roda os confrontos do debate eleitoral simulado")
    parser.add_argument("--personas", required=True, help="JSON de eleitores gerado por gerar_personas.py")
    parser.add_argument("--rodadas", type=int, default=5)
    parser.add_argument("--confrontos", nargs="+", type=_parse_confronto, default=None,
                        help="ex: emocional:logico logico:credibilidade (padrão: todos os pares)")
    parser.add_argument("--contrabalancear", action="store_true",
                        help="roda cada confronto também com os rótulos 1/2 invertidos")
    parser.add_argument("--repeticoes", type=int, default=1,
                        help="nº de repetições de cada confronto (a seed muda a cada uma)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modelo", default="llama3.1:8b")
    parser.add_argument("--temperatura", type=float, default=0.7)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--max-rodadas-historico", type=int, default=3,
                        help="rodadas anteriores mandadas como contexto (0 = debate inteiro)")
    parser.add_argument("--workers", type=int, default=4, help="eleitores avaliados em paralelo")
    parser.add_argument("--db", default="storage/experimento.db")
    args = parser.parse_args()

    eleitores = carregar_eleitores_json(args.personas)
    print(f"{len(eleitores)} eleitores carregados de {args.personas}")

    confrontos = args.confrontos or list(combinations(TIPOS, 2))
    if args.contrabalancear:
        confrontos += [(b, a) for a, b in confrontos]

    storage = Storage(args.db)
    try:
        for repeticao in range(args.repeticoes):
            config = ConfigLLM(modelo=args.modelo, temperatura=args.temperatura,
                               seed=args.seed + repeticao, num_ctx=args.num_ctx)
            llm = LLMClient(config)
            for id_1, id_2 in confrontos:
                run_id = uuid.uuid4().hex[:8]
                print(f"\n=== {id_1} x {id_2} | repetição {repeticao + 1}/{args.repeticoes} "
                      f"| seed {config.seed} | run {run_id} ===")
                storage.criar_run(run_id, id_1, id_2, args.rodadas, config.modelo,
                                  config.temperatura, config.seed, args.personas)
                storage.salvar_eleitores(run_id, eleitores)
                OrquestradorDebate(
                    run_id=run_id,
                    tipo_1=TIPOS[id_1],
                    tipo_2=TIPOS[id_2],
                    eleitores=eleitores,
                    llm=llm,
                    storage=storage,
                    max_rodadas_historico=args.max_rodadas_historico or None,
                    max_workers=args.workers,
                ).rodar(args.rodadas)
    finally:
        storage.close()

    print(f"\nConcluído. Dados em {args.db}")


if __name__ == "__main__":
    main()
