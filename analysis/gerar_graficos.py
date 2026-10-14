"""
Gera gráficos a partir dos dados de uma simulação já rodada (banco SQLite).
Pensado para a reunião/apresentação -- visual rápido, não a análise
estatística final do TCC (essa vem depois, com regressão e mais dados).

Uso:
    python -m analysis.gerar_graficos
    python -m analysis.gerar_graficos --db storage/debate_sim.db --out graficos/

Gera 4 imagens em --out:
    1. correlacao_tracos.png      -- correlação traço x nº de mudanças de voto
    2. evolucao_intencao_voto.png -- como a distribuição de voto evolui por rodada
    3. confianca_por_rodada.png   -- confiança média dos eleitores por rodada
    4. mudancas_por_estrategia.png-- nº de mudanças associadas a cada estratégia argumentativa
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from analysis.explorar import carregar_dataframes, resumo_mudancas_por_eleitor, correlacao_tracos_x_mudancas


PALETA = {
    "A": "#1D9E75",        # teal
    "B": "#D85A30",        # coral
    "indeciso": "#888780",  # gray
}


def plot_correlacao_tracos(tabela_resumo: pd.DataFrame, out_dir: Path):
    correlacao = correlacao_tracos_x_mudancas(tabela_resumo)

    fig, ax = plt.subplots(figsize=(8, 5))
    cores = ["#1D9E75" if v >= 0 else "#D85A30" for v in correlacao.values]
    ax.barh(correlacao.index, correlacao.values, color=cores)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Correlação com número de mudanças de voto")
    ax.set_title("Traços de personalidade/morais x mudança de intenção de voto")
    ax.set_xlim(-1, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "correlacao_tracos.png", dpi=150)
    plt.close(fig)
    print(f"  salvo: correlacao_tracos.png")


def plot_evolucao_intencao_voto(respostas: pd.DataFrame, out_dir: Path):
    """Para cada rodada, conta quantos eleitores estão em A / B / indeciso."""
    contagem = (
        respostas.groupby(["rodada", "intencao_voto"])
        .size()
        .unstack(fill_value=0)
    )
    # garante que todas as colunas existam, mesmo que uma opção nunca apareça
    for col in ["A", "B", "indeciso"]:
        if col not in contagem.columns:
            contagem[col] = 0
    contagem = contagem[["A", "B", "indeciso"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    for col in ["A", "B", "indeciso"]:
        ax.plot(contagem.index, contagem[col], marker="o", label=col, color=PALETA[col])
    ax.set_xlabel("Rodada")
    ax.set_ylabel("Número de eleitores")
    ax.set_title("Evolução da intenção de voto ao longo das rodadas")
    ax.set_xticks(contagem.index)
    ax.legend(title="Intenção de voto")
    fig.tight_layout()
    fig.savefig(out_dir / "evolucao_intencao_voto.png", dpi=150)
    plt.close(fig)
    print(f"  salvo: evolucao_intencao_voto.png")


def plot_confianca_por_rodada(respostas: pd.DataFrame, out_dir: Path):
    """A rodada 0 (estado inicial, antes do debate) é excluída aqui --
    sua confiança é um placeholder fixo (0.0), não uma confiança real
    avaliada pelo modelo, então não faz sentido entrar nessa média."""
    respostas_debate = respostas[respostas["rodada"] > 0]
    media_por_rodada = respostas_debate.groupby("rodada")["confianca"].mean()
    desvio_por_rodada = respostas_debate.groupby("rodada")["confianca"].std().fillna(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        media_por_rodada.index, media_por_rodada.values,
        yerr=desvio_por_rodada.values, marker="o", capsize=4, color="#534AB7",
    )
    ax.set_xlabel("Rodada")
    ax.set_ylabel("Confiança média (0-1)")
    ax.set_title("Confiança média dos eleitores por rodada (± desvio padrão)")
    ax.set_xticks(media_por_rodada.index)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "confianca_por_rodada.png", dpi=150)
    plt.close(fig)
    print(f"  salvo: confianca_por_rodada.png")


def plot_mudancas_por_estrategia(respostas: pd.DataFrame, falas: pd.DataFrame, out_dir: Path):
    """
    Aproximação simples para RQ3: para cada rodada, identifica as
    estratégias usadas pelos candidatos (pode haver mais de uma -- 4 falas
    por rodada) e associa ao número de mudanças de voto registradas
    naquela rodada. Como cada rodada tem várias falas/estratégias
    possivelmente diferentes, este gráfico é só exploratório -- uma
    análise causal de fato exigiria isolar 1 estratégia por rodada
    (ex: fixando a estratégia em vez de sortear, como o código já permite).
    """
    mudancas_por_rodada = respostas.groupby("rodada")["mudou_de_posicao"].sum()

    # para cada rodada, lista as estratégias usadas (pode repetir)
    estrategias_por_rodada = falas.groupby("rodada")["estrategia"].apply(list)

    linhas = []
    for rodada, estrategias in estrategias_por_rodada.items():
        n_mudancas = mudancas_por_rodada.get(rodada, 0)
        for estrategia in set(estrategias):  # cada estratégia usada na rodada conta 1x
            linhas.append({"rodada": rodada, "estrategia": estrategia, "n_mudancas": n_mudancas})

    df_plot = pd.DataFrame(linhas)
    if df_plot.empty:
        print("  aviso: sem dados de estratégia para plotar, pulando mudancas_por_estrategia.png")
        return

    media_por_estrategia = df_plot.groupby("estrategia")["n_mudancas"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(media_por_estrategia.index, media_por_estrategia.values, color="#3C3489")
    ax.set_ylabel("Mudanças de voto médias na rodada")
    ax.set_title(
        "Estratégia argumentativa x mudanças de voto (exploratório)\n"
        "média de mudanças nas rodadas em que a estratégia foi usada",
        fontsize=11,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "mudancas_por_estrategia.png", dpi=150)
    plt.close(fig)
    print(f"  salvo: mudancas_por_estrategia.png")


def plot_ranking_eleitores(tabela_resumo: pd.DataFrame, out_dir: Path):
    """Ranking simples de eleitores por número de mudanças de voto --
    fácil de ler individualmente, bom para apontar exemplos específicos
    na apresentação (ex: 'olha esse eleitor mudou 3x, tem abertura alta')."""
    dados = tabela_resumo.sort_values("n_mudancas", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(dados))))
    ax.barh(dados["eleitor_id"], dados["n_mudancas"], color="#1D9E75")
    ax.set_xlabel("Número de mudanças de intenção de voto")
    ax.set_title("Mudanças de voto por eleitor")
    fig.tight_layout()
    fig.savefig(out_dir / "ranking_mudancas_por_eleitor.png", dpi=150)
    plt.close(fig)
    print(f"  salvo: ranking_mudancas_por_eleitor.png")


def main():
    parser = argparse.ArgumentParser(description="Gera gráficos a partir do banco de dados da simulação")
    parser.add_argument("--db", type=str, default="storage/debate_sim.db")
    parser.add_argument("--out", type=str, default="graficos")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Carregando dados de {args.db}...")
    eleitores, respostas, falas = carregar_dataframes(args.db)
    tabela_resumo = resumo_mudancas_por_eleitor(eleitores, respostas)

    print(f"Gerando gráficos em {out_dir}/...")
    plot_correlacao_tracos(tabela_resumo, out_dir)
    plot_ranking_eleitores(tabela_resumo, out_dir)
    plot_evolucao_intencao_voto(respostas, out_dir)
    plot_confianca_por_rodada(respostas, out_dir)
    plot_mudancas_por_estrategia(respostas, falas, out_dir)

    print("\nConcluído.")


if __name__ == "__main__":
    main()
