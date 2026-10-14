"""
Análise exploratória simples: junta eleitores + respostas e olha a
relação entre traços (Big Five / Fundamentos Morais) e mudança de
intenção de voto ao longo das rodadas.

Isso é DELIBERADAMENTE simples (estatística descritiva, não inferencial)
-- o objetivo aqui é ter algo visual para mostrar na reunião, não a
análise final do TCC. A análise inferencial (regressão logística/mista
para RQ1/RQ2) vem depois, com volume de dados maior.
"""

import sqlite3
import json
import pandas as pd


def carregar_dataframes(db_path: str = "storage/debate_sim.db"):
    conn = sqlite3.connect(db_path)

    eleitores_raw = pd.read_sql("SELECT * FROM eleitores", conn)
    respostas = pd.read_sql("SELECT * FROM respostas", conn)
    falas = pd.read_sql("SELECT * FROM falas", conn)
    conn.close()

    # "Achata" os campos JSON de big_five e fundamentos_morais em colunas
    big_five_df = pd.json_normalize(eleitores_raw["big_five_json"].apply(json.loads))
    fundamentos_df = pd.json_normalize(eleitores_raw["fundamentos_morais_json"].apply(json.loads))
    perfil_df = pd.json_normalize(eleitores_raw["perfil_json"].apply(json.loads))

    eleitores = pd.concat(
        [eleitores_raw[["run_id", "eleitor_id", "narrativa"]], perfil_df, big_five_df, fundamentos_df],
        axis=1,
    )

    return eleitores, respostas, falas


def resumo_mudancas_por_eleitor(eleitores: pd.DataFrame, respostas: pd.DataFrame) -> pd.DataFrame:
    """Quantas vezes cada eleitor mudou de posição ao longo da simulação,
    cruzado com seus traços -- a tabela mais direta para RQ1/RQ2.

    A rodada 0 (estado inicial "indeciso" antes do debate, confiança
    placeholder 0.0) é excluída do cálculo de confiança média -- não é
    uma confiança real avaliada pelo modelo. `mudou_de_posicao` na
    rodada 0 é sempre False por construção, então não precisa de
    tratamento especial ali."""
    respostas_debate = respostas[respostas["rodada"] > 0]
    mudancas = respostas.groupby("eleitor_id")["mudou_de_posicao"].sum().rename("n_mudancas")
    confianca_media = respostas_debate.groupby("eleitor_id")["confianca"].mean().rename("confianca_media")

    tabela = eleitores.merge(mudancas, on="eleitor_id", how="left")
    tabela = tabela.merge(confianca_media, on="eleitor_id", how="left")
    tabela["n_mudancas"] = tabela["n_mudancas"].fillna(0)

    colunas_interesse = [
        "eleitor_id", "idade", "regiao", "n_mudancas", "confianca_media",
        "abertura", "conscienciosidade", "extroversao", "amabilidade", "neuroticismo",
        "cuidado_dano", "justica_reciprocidade", "lealdade_grupo",
        "autoridade_respeito", "pureza_santidade",
    ]
    return tabela[colunas_interesse].sort_values("n_mudancas", ascending=False)


def correlacao_tracos_x_mudancas(tabela_resumo: pd.DataFrame) -> pd.Series:
    """Correlação simples (Pearson) entre cada traço e número de mudanças.
    Com N pequeno (protótipo) isso é só exploratório -- não tem poder
    estatístico para conclusões, mas já mostra direção e serve de
    sanity-check de que os traços estão de fato gerando variação no
    comportamento simulado."""
    tracos = [
        "abertura", "conscienciosidade", "extroversao", "amabilidade", "neuroticismo",
        "cuidado_dano", "justica_reciprocidade", "lealdade_grupo",
        "autoridade_respeito", "pureza_santidade",
    ]
    return tabela_resumo[tracos + ["n_mudancas"]].corr()["n_mudancas"].drop("n_mudancas").sort_values(ascending=False)


if __name__ == "__main__":
    eleitores, respostas, falas = carregar_dataframes()
    tabela = resumo_mudancas_por_eleitor(eleitores, respostas)
    print("Resumo por eleitor:")
    print(tabela.to_string(index=False))

    print("\nCorrelação (traço x número de mudanças de posição):")
    print(correlacao_tracos_x_mudancas(tabela))
