"""
Persistência em SQLite. Um banco guarda o experimento inteiro: cada
confronto/repetição é uma linha em `runs`, e todas as outras tabelas
carregam `run_id` (sempre cruzar por ele na análise).
"""

from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

from schemas import Fala, PersonaEleitor, RespostaEleitor


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    tipo_candidato_1  TEXT NOT NULL,
    tipo_candidato_2  TEXT NOT NULL,
    n_rodadas         INTEGER,
    modelo_llm        TEXT,
    temperatura       REAL,
    seed              INTEGER,
    arquivo_personas  TEXT,
    criado_em         TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS eleitores (
    run_id                   TEXT,
    eleitor_id               TEXT,
    perfil_json              TEXT,
    big_five_json            TEXT,
    fundamentos_morais_json  TEXT,
    narrativa                TEXT,
    PRIMARY KEY (run_id, eleitor_id)
);

CREATE TABLE IF NOT EXISTS falas (
    run_id          TEXT,
    rodada          INTEGER,
    ordem           INTEGER,
    orador          TEXT,
    tipo_candidato  TEXT,
    tipo_fala       TEXT,
    tema            TEXT,
    texto           TEXT
);

CREATE TABLE IF NOT EXISTS respostas (
    run_id            TEXT,
    rodada            INTEGER,
    eleitor_id        TEXT,
    intencao_voto     TEXT,
    confianca         REAL,
    justificativa     TEXT,
    mudou_de_posicao  INTEGER
);
"""


class Storage:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(SCHEMA)

    def criar_run(self, run_id: str, tipo_1: str, tipo_2: str, n_rodadas: int,
                  modelo_llm: str, temperatura: float, seed: int | None, arquivo_personas: str):
        with self.conn:
            self.conn.execute(
                "INSERT INTO runs (run_id, tipo_candidato_1, tipo_candidato_2, n_rodadas, "
                "modelo_llm, temperatura, seed, arquivo_personas) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, tipo_1, tipo_2, n_rodadas, modelo_llm, temperatura, seed, arquivo_personas),
            )

    def salvar_eleitores(self, run_id: str, eleitores: list[PersonaEleitor]):
        with self.conn:
            self.conn.executemany(
                "INSERT INTO eleitores VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (run_id, p.id,
                     json.dumps(asdict(p.perfil), ensure_ascii=False),
                     json.dumps(asdict(p.big_five), ensure_ascii=False),
                     json.dumps(asdict(p.fundamentos_morais), ensure_ascii=False),
                     p.narrativa)
                    for p in eleitores
                ],
            )

    def salvar_fala(self, run_id: str, f: Fala):
        with self.conn:
            self.conn.execute(
                "INSERT INTO falas VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, f.rodada, f.ordem, f.orador, f.tipo_candidato, f.tipo_fala, f.tema, f.texto),
            )

    def salvar_resposta(self, run_id: str, r: RespostaEleitor):
        with self.conn:
            self.conn.execute(
                "INSERT INTO respostas VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, r.rodada, r.eleitor_id, r.intencao_voto, r.confianca,
                 r.justificativa, int(r.mudou_de_posicao)),
            )

    def close(self):
        self.conn.close()
