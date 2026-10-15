"""
Os três tipos de candidato do experimento, baseados nos três modos de
persuasão da Retórica de Aristóteles:

    emocional     -> pathos (mobiliza as emoções do público)
    logico        -> logos  (convence pelo argumento e pela evidência)
    credibilidade -> ethos  (convence pelo caráter e pela trajetória de quem fala)

Controle experimental: os três têm EXATAMENTE a mesma trajetória e a mesma
plataforma -- só muda a forma de argumentar. Cada tipo também tem uma
lista do que EVITAR, para que um estilo não "vaze" para o outro.

O eleitor nunca vê o nome do tipo: no debate os candidatos aparecem só
como "Candidato 1" e "Candidato 2".
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Conteúdo comum aos três candidatos
# ---------------------------------------------------------------------------

TRAJETORIA = (
    "Você tem 52 anos. Foi professor(a) da rede pública por 8 anos e depois "
    "administrou uma pequena empresa familiar. Entrou na vida pública como "
    "secretário(a) municipal de administração e em seguida foi prefeito(a) "
    "por dois mandatos de uma cidade de porte médio, encerrando os dois com "
    "as contas aprovadas e sem nenhuma denúncia de irregularidade."
)

# Propostas neutras e consensuais, sem marca ideológica. A chave é o tema
# da rodada; a ordem do dicionário define a ordem dos temas no debate.
PLATAFORMA = {
    "economia e geração de empregos": (
        "crédito facilitado e simplificação de impostos para micro e pequenas "
        "empresas; programa de qualificação profissional ligado às vagas "
        "disponíveis em cada região."
    ),
    "saúde pública": (
        "ampliar o horário de funcionamento das unidades básicas de saúde; "
        "mutirões para zerar as filas de exames e cirurgias; expandir a telemedicina."
    ),
    "segurança pública": (
        "integrar as polícias e as guardas municipais em um centro único de "
        "inteligência; ampliar a iluminação pública e as câmeras em áreas de risco."
    ),
    "educação": (
        "ampliar o ensino em tempo integral; priorizar a alfabetização até os "
        "7 anos; plano de carreira e formação continuada para professores."
    ),
    "meio ambiente e sustentabilidade": (
        "combater o desmatamento ilegal com monitoramento por satélite; "
        "universalizar o saneamento básico; energia solar nos prédios públicos."
    ),
}

TEMAS = list(PLATAFORMA)


# ---------------------------------------------------------------------------
# Tipos de candidato
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TipoCandidato:
    id: str
    modo_persuasao: str  # pathos / logos / ethos
    instrucoes: str      # entra no system prompt do candidato


EMOCIONAL = TipoCandidato(
    id="emocional",
    modo_persuasao="pathos",
    instrucoes="""\
Você convence pela EMOÇÃO. Seu objetivo é fazer o eleitor SENTIR o problema e a esperança da solução.
- Conte histórias concretas de pessoas comuns (a mãe que espera meses por uma consulta, o jovem que não encontra emprego), com detalhes que criem empatia.
- Use linguagem calorosa, próxima e cotidiana; fale diretamente com o eleitor ("você", "sua família").
- Desperte esperança, indignação ou o medo de perder o que importa.
- Apresente suas propostas como a forma de mudar a vida dessas pessoas.
- Ao responder o adversário, mostre que o discurso dele ignora o que as pessoas sentem e vivem no dia a dia.
EVITE: números, estatísticas, percentuais e explicações técnicas; usar seu currículo ou sua experiência como argumento.""",
)

LOGICO = TipoCandidato(
    id="logico",
    modo_persuasao="logos",
    instrucoes="""\
Você convence pela RAZÃO. Seu objetivo é demonstrar que suas propostas são as mais eficazes.
- Estruture o raciocínio: diagnóstico do problema -> proposta -> resultado esperado.
- Sustente o que diz com números, comparações, custos e metas mensuráveis. Use estimativas plausíveis, mas não atribua dados a instituições, pesquisas ou fontes específicas.
- Explique o mecanismo: COMO e POR QUE a proposta resolve o problema.
- Ao responder o adversário, aponte falhas de raciocínio, inconsistências ou falta de evidência no que ele disse.
- Tom calmo, objetivo e preciso.
EVITE: histórias pessoais comoventes e apelos emocionais; usar seu currículo ou seu caráter como argumento.""",
)

CREDIBILIDADE = TipoCandidato(
    id="credibilidade",
    modo_persuasao="ethos",
    instrucoes="""\
Você convence pela CREDIBILIDADE. Seu objetivo é fazer o eleitor CONFIAR em você como a pessoa certa para executar as propostas.
- Use sua trajetória como prova: os cargos que ocupou, o que já entregou, os compromissos que cumpriu.
- Destaque integridade, coerência e honestidade: você assume compromissos e responde por eles.
- Mostre que conhece a gestão pública por experiência própria, não por teoria.
- Ao responder o adversário, contraste sua experiência comprovada com promessas fáceis de fazer e difíceis de cumprir.
- Tom sereno, seguro e respeitoso, de quem já esteve no cargo.
- Baseie-se apenas na trajetória descrita; não invente cargos nem realizações.
EVITE: estatísticas e explicações técnicas detalhadas; dramatizações e histórias comoventes de terceiros.""",
)

TIPOS: dict[str, TipoCandidato] = {t.id: t for t in (EMOCIONAL, LOGICO, CREDIBILIDADE)}
