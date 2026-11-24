"""Catálogo explícito de indicadores.

Cada indicador corresponde a uma tipologia da Carta Circular nº 4.001/2020
(ou ao resultado do KYC externo) e entra no score com o peso declarado aqui.
O bloqueio interno BLOCKED_BY_RISK não é indicador e não altera o score.
"""

from __future__ import annotations

INDICATORS: dict[str, dict] = {
    "CASH_STRUCTURING": {
        "title": "Fragmentação de saques",
        "short": "Saques fracionados",
        "weight": 26,
        "carta": (
            "Fragmentação de saques em espécie, a fim de burlar limites regulatórios de reporte."
        ),
        "rule": (
            "Quatro ou mais saques em espécie em até 6 horas, com valores próximos "
            "(diferença entre o maior e o menor de até 35% da média) e soma de pelo menos R$ 2.000."
        ),
    },
    "MULTIPLE_PIX_SHORT_PERIOD": {
        "title": "Múltiplos PIX em curto período",
        "short": "Rajada de PIX",
        "weight": 20,
        "carta": (
            "Aumento no volume de transferências, sem causa aparente, concentrado em curto período."
        ),
        "rule": "Dez ou mais PIX efetivados em qualquer janela de 3 horas.",
    },
    "RAPID_IN_OUT": {
        "title": "Entrada e saída rápida",
        "short": "Passagem rápida",
        "weight": 22,
        "carta": "Recebimento de créditos com o imediato débito dos valores.",
        "rule": (
            "Crédito de pelo menos R$ 1.500 e de meia renda mensal, seguido em até 6 horas "
            "de débitos por PIX, TED ou saque que somam 75% ou mais do valor recebido."
        ),
    },
    "HIGH_VOLUME_TO_INCOME": {
        "title": "Volume incompatível com a renda",
        "short": "Volume x renda",
        "weight": 24,
        "carta": (
            "Movimentação de recursos incompatível com o patrimônio, a atividade econômica "
            "ou a ocupação profissional e a capacidade financeira."
        ),
        "rule": (
            "Crédito igual ou superior a 8 rendas mensais, com saída de pelo menos metade "
            "em até 48 horas por PIX, TED ou saque; ou débitos efetivados, em qualquer "
            "janela de 7 dias, iguais ou superiores a 3 rendas mensais."
        ),
    },
    "UNUSUAL_HOUR": {
        "title": "Movimentação relevante de madrugada",
        "short": "Horário atípico",
        "weight": 18,
        "carta": "Transferências ou saques com atipicidade, aqui observada na combinação de horário e valor.",
        "rule": (
            "Débito por PIX, TED ou saque entre 00h e 05h, de valor igual ou superior a 2 rendas mensais."
        ),
    },
    "MANY_ORIGINATORS": {
        "title": "Muitas origens em curto período",
        "short": "Várias origens",
        "weight": 24,
        "carta": (
            "Conta que recebe depósitos ou transferências de várias origens, em curto período, "
            "situação associada à dispersão da origem dos recursos."
        ),
        "rule": "Doze ou mais contas de origem distintas creditando PIX ou TED em até 72 horas.",
    },
    "REPEATED_EXTERNAL": {
        "title": "Envios repetidos à mesma conta externa",
        "short": "Destino recorrente",
        "weight": 22,
        "carta": (
            "Transferências recorrentes, em curto intervalo, para um mesmo destino sem relação "
            "evidente com a capacidade financeira declarada."
        ),
        "rule": (
            "Quatro ou mais débitos por PIX ou TED para a mesma conta externa em até 5 dias, "
            "somando pelo menos R$ 1.500 ou 35% da renda mensal, o que for maior."
        ),
    },
    "ADVERSE_KYC": {
        "title": "KYC externo adverso",
        "short": "KYC adverso",
        "weight": 42,
        "carta": "Apontamento de fonte externa: sanção, risco de PLD ou suspeita de fraude.",
        "rule": (
            "Bureau de KYC com status bloqueado e motivo SANCTIONS_MATCH, AML_RISK ou SUSPECTED_FRAUD. "
            "O motivo interno BLOCKED_BY_RISK não dispara este indicador."
        ),
    },
}

ADVERSE_KYC_REASONS = ("SANCTIONS_MATCH", "AML_RISK", "SUSPECTED_FRAUD")

KYC_REASON_LABEL = {
    "SANCTIONS_MATCH": "coincidência com lista de sanções",
    "AML_RISK": "risco de PLD indicado pelo bureau",
    "SUSPECTED_FRAUD": "suspeita de fraude indicada pelo bureau",
    "CPF_IRREGULAR": "CPF irregular",
    "CPF_INACTIVE": "CPF inativo",
    "CNPJ_IRREGULAR": "CNPJ irregular",
    "CNPJ_INACTIVE": "CNPJ inativo",
    "DECEASED_HOLDER": "titular falecido",
    "COMPANY_CLOSED": "empresa baixada",
    "COMPANY_CANCELLED": "empresa cancelada",
    "DOCUMENT_NOT_FOUND": "documento não localizado",
    "DOCUMENT_DATA_MISMATCH": "divergência cadastral",
    "UNDERAGE": "menor de idade",
    "BLOCKED_BY_RISK": "bloqueio interno por risco",
}

LEVEL_LABEL = {
    "critico": "Crítico",
    "alto": "Alto",
    "medio": "Médio",
    "sem_alerta": "Sem alerta",
}

# Soma dos pesos. As faixas são transparentes para o analista.
LEVEL_RULES = (
    (60, "critico"),
    (36, "alto"),
    (20, "medio"),
)


def score_of(codes: list[str]) -> int:
    return min(100, sum(INDICATORS[code]["weight"] for code in codes if code in INDICATORS))


def level_of(score: int, findings: int) -> str:
    if findings <= 0 or score <= 0:
        return "sem_alerta"
    for minimum, name in LEVEL_RULES:
        if score >= minimum:
            return name
    return "medio"
