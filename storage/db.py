"""
Persistência em SQLite. Schema simples, pensado para facilitar a
exportação direta para pandas na hora da análise (cada tabela ~ um
dataframe).

Tabelas:
- runs: cada execução completa da simulação (importante p/ RQ4)
- eleitores: persona completa de cada eleitor (serializada em JSON
  nas colunas de traços, para não explodir em N colunas)
- candidatos: persona dos candidatos
- falas: cada fala de candidato por rodada
- respostas: cada resposta de eleitor por rodada (a tabela "principal"
  para a análise estatística)
"""

import sqlite3
import json
from pathlib import Path

from schemas import PersonaEleitor, PersonaCandidato, FalaCandidato, RespostaEleitor


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    descricao TEXT,
    modelo_llm TEXT,
    seed INTEGER,
    criado_em TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS eleitores (
    run_id TEXT,
    eleitor_id TEXT,
    perfil_json TEXT,
    big_five_json TEXT,
    fundamentos_morais_json TEXT,
    narrativa TEXT,
    PRIMARY KEY (run_id, eleitor_id)
);

CREATE TABLE IF NOT EXISTS candidatos (
    run_id TEXT,
    candidato_id TEXT,
    nome TEXT,
    plataforma TEXT,
    estilo_retorico TEXT,
    PRIMARY KEY (run_id, candidato_id)
);

CREATE TABLE IF NOT EXISTS falas (
    run_id TEXT,
    rodada INTEGER,
    candidato_id TEXT,
    estrategia TEXT,
    texto TEXT,
    tipo_fala TEXT,
    tema TEXT,
    pergunta_moderador TEXT
);

CREATE TABLE IF NOT EXISTS respostas (
    run_id TEXT,
    rodada INTEGER,
    eleitor_id TEXT,
    intencao_voto TEXT,
    confianca REAL,
    justificativa TEXT,
    mudou_de_posicao INTEGER
);
"""


class Storage:
    def __init__(self, db_path: str = "debate_sim.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def criar_run(self, run_id: str, descricao: str, modelo_llm: str, seed: int | None):
        self.conn.execute(
            "INSERT INTO runs (run_id, descricao, modelo_llm, seed) VALUES (?, ?, ?, ?)",
            (run_id, descricao, modelo_llm, seed),
        )
        self.conn.commit()

    def salvar_eleitor(self, run_id: str, persona: PersonaEleitor):
        self.conn.execute(
            """INSERT OR REPLACE INTO eleitores
               (run_id, eleitor_id, perfil_json, big_five_json, fundamentos_morais_json, narrativa)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                run_id, persona.id,
                json.dumps(persona.perfil.to_dict(), ensure_ascii=False),
                json.dumps(persona.big_five.to_dict(), ensure_ascii=False),
                json.dumps(persona.fundamentos_morais.to_dict(), ensure_ascii=False),
                persona.narrativa,
            ),
        )
        self.conn.commit()

    def salvar_candidato(self, run_id: str, persona: PersonaCandidato):
        self.conn.execute(
            """INSERT OR REPLACE INTO candidatos
               (run_id, candidato_id, nome, plataforma, estilo_retorico)
               VALUES (?, ?, ?, ?, ?)""",
            (run_id, persona.id, persona.nome, persona.plataforma, persona.estilo_retorico),
        )
        self.conn.commit()

    def salvar_fala(self, run_id: str, fala: FalaCandidato):
        self.conn.execute(
            """INSERT INTO falas
               (run_id, rodada, candidato_id, estrategia, texto, tipo_fala, tema, pergunta_moderador)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id, fala.rodada, fala.candidato_id, fala.estrategia, fala.texto,
                fala.tipo_fala, fala.tema, fala.pergunta_moderador,
            ),
        )
        self.conn.commit()

    def salvar_resposta(self, resposta: RespostaEleitor):
        self.conn.execute(
            """INSERT INTO respostas
               (run_id, rodada, eleitor_id, intencao_voto, confianca, justificativa, mudou_de_posicao)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                resposta.run_id, resposta.rodada, resposta.eleitor_id,
                resposta.intencao_voto, resposta.confianca,
                resposta.justificativa, int(resposta.mudou_de_posicao),
            ),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
