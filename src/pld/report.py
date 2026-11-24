"""Redação do dossiê a partir dos fatos calculados pelo motor de regras.

O texto só menciona números presentes nos fatos. Não completa lacunas com
notícias, cidades de saque ou relações que a base não traz.
"""

from __future__ import annotations

from .format import brl, duration_label, ratio_label, status_label
from .indicators import INDICATORS, KYC_REASON_LABEL, LEVEL_LABEL, level_of, score_of

TYPE_LABEL = {
    "pix": "PIX",
    "ted": "TED",
    "card": "cartão",
    "cash_withdrawal": "saque em espécie",
}
DIR_LABEL = {"debit": "débito", "credit": "crédito"}


def reason_label(code: str | None) -> str:
    if not code:
        return "sem motivo registrado"
    return KYC_REASON_LABEL.get(code, code.replace("_", " ").lower())


def _examples_sentence(examples: list[dict], limit: int = 4) -> str:
    parts = []
    for item in examples[:limit]:
        kind = TYPE_LABEL.get(item["type"], item["type"])
        direction = DIR_LABEL.get(item["direction"], item["direction"])
        parts.append(
            f"{item['when']}, {kind} {direction} de {brl(item['amount'])} ({item['detail']})"
        )
    if not parts:
        return ""
    return " Citamos como exemplos: " + "; ".join(parts) + "."


def _occupation(profile: dict) -> str:
    if profile.get("document_type") == "CNPJ":
        return profile.get("company_name") or "pessoa jurídica"
    return profile.get("occupation") or "ocupação não informada"


def describe(code: str, facts: dict, profile: dict) -> tuple[str, str]:
    income = profile["income"]
    who = _occupation(profile)
    examples = facts.get("examples") or []

    if code == "CASH_STRUCTURING":
        summary = (
            f"{facts['count']} saques em {duration_label(facts['hours'])}, "
            f"somando {brl(facts['total'])}."
        )
        narrative = (
            "Identificamos a prática atípica de saques efetuados em espécie, de valores próximos "
            f"entre si, entre {facts['start']} e {facts['end']} "
            f"({duration_label(facts['hours'])}). Foram {facts['count']} saques, "
            f"totalizando {brl(facts['total'])}. A fragmentação dificulta a identificação da "
            "destinação final dos recursos e se aproxima do fracionamento usado para contornar "
            f"limites de reporte. A renda mensal declarada é {brl(income)} ({who})."
            + _examples_sentence(examples)
        )
        return summary, narrative

    if code == "MULTIPLE_PIX_SHORT_PERIOD":
        summary = (
            f"{facts['count']} PIX em {duration_label(facts['hours'])}, "
            f"somando {brl(facts['total'])}."
        )
        narrative = (
            f"Entre {facts['start']} e {facts['end']}, em {duration_label(facts['hours'])}, "
            f"o analisado movimentou {facts['count']} PIX, no total de {brl(facts['total'])}. "
            "A concentração nesse intervalo não encontra correspondência imediata com a renda "
            f"mensal declarada de {brl(income)} ({who})."
            + _examples_sentence(examples)
        )
        return summary, narrative

    if code == "RAPID_IN_OUT":
        summary = (
            f"Crédito de {brl(facts['credit_amount'])} e saída de {facts['out_pct']}% "
            f"em {duration_label(facts['out_hours'])}."
        )
        narrative = (
            f"Em {facts['credit_when']} houve crédito de {brl(facts['credit_amount'])} "
            f"por {TYPE_LABEL.get(facts['credit_type'], facts['credit_type'])} "
            f"({facts['credit_detail']}). Em seguida, {facts['out_count']} débito(s) por PIX, TED "
            f"ou saque somaram {brl(facts['out_total'])}, o equivalente a {facts['out_pct']}% "
            f"do valor recebido, em {duration_label(facts['out_hours'])}. O padrão é de passagem "
            "rápida: o crédito entra e deixa a conta antes de permanecer como disponibilidade do cliente."
            + _examples_sentence(examples)
        )
        return summary, narrative

    if code == "HIGH_VOLUME_TO_INCOME":
        pieces = []
        bits = []
        if facts.get("credit_amount"):
            arrived = (
                f"Em {facts['credit_when']} entrou {brl(facts['credit_amount'])} por "
                f"{TYPE_LABEL.get(facts['credit_type'], facts['credit_type'])} "
                f"({facts['credit_detail']}), valor equivalente a {ratio_label(facts['credit_ratio'])} "
                f"vezes a renda mensal de {brl(income)}."
            )
            if facts.get("out_count"):
                pieces.append(
                    f"crédito de {brl(facts['credit_amount'])} ({ratio_label(facts['credit_ratio'])} rendas) "
                    f"com saída de {facts['out_pct']}% em até 48 horas"
                )
                bits.append(
                    arrived + (
                        f" Nas 48 horas seguintes, {facts['out_count']} débito(s) por PIX, TED ou saque "
                        f"somaram {brl(facts['out_total'])} ({facts['out_pct']}% do crédito)."
                    )
                )
            else:
                pieces.append(
                    f"crédito de {brl(facts['credit_amount'])} ({ratio_label(facts['credit_ratio'])} rendas), "
                    "acima da capacidade declarada"
                )
                bits.append(
                    arrived + " Não houve saída proporcional imediata; o valor recebido, por si só, destoa da renda declarada."
                )
        if facts.get("ratio_7d") and facts["ratio_7d"] >= 3:
            pieces.append(
                f"maior semana de débitos = {brl(facts['volume_7d'])} "
                f"({ratio_label(facts['ratio_7d'])} vezes a renda)"
            )
            bits.append(
                f"Na janela de maior intensidade, os débitos efetivados entre {facts['window_start']} "
                f"e {facts['window_end']} somaram {brl(facts['volume_7d'])}, "
                f"{ratio_label(facts['ratio_7d'])} vezes a renda mensal declarada ({who})."
            )
        summary = "; ".join(pieces) if pieces else "Volume acima da capacidade financeira declarada."
        summary = summary[0].upper() + summary[1:]
        narrative = " ".join(bits) + (
            " A movimentação é incompatível, na comparação direta, com a capacidade financeira informada no cadastro."
            + _examples_sentence(examples)
        )
        return summary, narrative

    if code == "UNUSUAL_HOUR":
        summary = (
            f"{facts['count']} débito(s) entre 00h e 05h, o maior de {brl(facts['max_amount'])} "
            f"({ratio_label(facts['max_ratio'])} rendas)."
        )
        narrative = (
            f"Entre 00h e 05h, localizamos {facts['count']} débito(s) por PIX, TED ou saque "
            f"de valor relevante, somando {brl(facts['total'])}. O maior foi de {brl(facts['max_amount'])} "
            f"em {facts['max_when']}, equivalente a {ratio_label(facts['max_ratio'])} vezes a renda "
            f"mensal de {brl(income)} ({who}). O horário, associado ao valor, foge do uso cotidiano "
            "observado para o perfil."
            + _examples_sentence(examples)
        )
        return summary, narrative

    if code == "MANY_ORIGINATORS":
        summary = (
            f"{facts['originators']} origens distintas em {duration_label(facts['hours'])}, "
            f"creditando {brl(facts['total'])}."
        )
        narrative = (
            f"Entre {facts['start']} e {facts['end']} ({duration_label(facts['hours'])}), "
            f"a conta recebeu {facts['count']} créditos por PIX ou TED vindos de "
            f"{facts['originators']} contas de origem distintas, no total de {brl(facts['total'])}. "
            "A dispersão de origens em intervalo curto é um sinal clássico de ocultação da "
            f"procedência dos recursos. Renda mensal declarada: {brl(income)} ({who})."
            + _examples_sentence(examples)
        )
        return summary, narrative

    if code == "REPEATED_EXTERNAL":
        summary = (
            f"{facts['count']} envios para a conta {facts['account']} em "
            f"{duration_label(facts['hours'])}, somando {brl(facts['total'])}."
        )
        narrative = (
            f"Entre {facts['start']} e {facts['end']}, o analisado enviou {facts['count']} PIX/TED "
            f"para a mesma conta externa {facts['account']}, totalizando {brl(facts['total'])} "
            f"em {duration_label(facts['hours'])}. A repetição do destino, no intervalo e no valor "
            f"frente à renda de {brl(income)} ({who}), pede verificação de quem é o beneficiário "
            "e qual a causa econômica dos envios."
            + _examples_sentence(examples)
        )
        return summary, narrative

    if code == "ADVERSE_KYC":
        label = reason_label(facts.get("reason"))
        summary = f"Bureau bloqueou o KYC: {label}."
        narrative = (
            "Em consulta à fonte externa de KYC, o cadastro consta bloqueado, "
            f"com o motivo {label} (código {facts.get('reason')}), "
            f"atualizado em {facts.get('updated') or 'data não informada'}. "
            "O apontamento independe da movimentação e, sozinho, justifica prioridade de leitura "
            "pelo analista. Listas e bureaus podem conter homônimo ou defasagem: a confirmação "
            "continua humana."
        )
        return summary, narrative

    return code, code


def identification(profile: dict) -> str:
    if profile["document_type"] == "CNPJ":
        owner = profile.get("owner_cpf") or "não informado"
        activity = profile.get("occupation") or "atividade do responsável não informada"
        return (
            f"Em consulta ao cadastro interno, consta a empresa {profile.get('company_name') or 'sem razão social'}, "
            f"CNPJ {profile['document']}, {activity}. Responsável cadastrado sob o CPF {owner}. "
            f"Faturamento declarado de {brl(profile['income'])} ao mês e patrimônio de {brl(profile['net_worth'])}. "
            f"Cadastro aberto em {profile['created_label']}. Domicílio informado em {profile['city']}/{profile['state']}."
        )
    born = profile.get("birth_label")
    age = profile.get("age_label")
    birth = f" Nascimento em {born}" + (f" ({age})." if age else ".")
    return (
        f"Em consulta ao cadastro interno, consta {profile['name']}, CPF {profile['document']}, "
        f"ocupação {profile.get('occupation') or 'não informada'}.{birth} "
        f"Renda declarada de {brl(profile['income'])} ao mês e patrimônio de {brl(profile['net_worth'])}. "
        f"Cadastro aberto em {profile['created_label']}. Domicílio informado em {profile['city']}/{profile['state']}."
    )


def recommendation(case: dict) -> str:
    level = case["level"]
    if level == "critico":
        return (
            "Recomendação de fila: priorizar a análise humana. Os sinais são convergentes e o caso "
            "comporta avaliação de comunicação ao COAF se o analista confirmar a atipicidade e a "
            "ausência de explicação econômica. Esta recomendação não comunica nada e não encerra a conta."
        )
    if level == "alto":
        return (
            "Recomendação de fila: aprofundar antes dos casos de score menor. Há sinal relevante, "
            "mas a decisão de comunicar, monitorar ou encerrar é do analista, com o contexto que a ferramenta não vê."
        )
    if level == "medio":
        return (
            "Recomendação de fila: revisar em lote, depois dos casos altos e críticos. O sinal existe "
            "e pode ser explicado por um evento lícito; o relatório serve para essa checagem ser rápida."
        )
    return (
        "Nenhum indicador do catálogo foi acionado no período. A ausência de alerta não é atestado "
        "de regularidade: cobre apenas as regras implementadas neste protótipo."
    )


def render_report(case: dict) -> str:
    profile = case["profile"]
    metrics = case["metrics"]
    codes = [finding["code"] for finding in case["findings"]]
    score = score_of(codes)
    level = level_of(score, len(codes))
    weights = ", ".join(
        f"{INDICATORS[code]['title']} ({INDICATORS[code]['weight']})" for code in codes
    ) or "nenhum indicador"

    lines = [
        f"# Dossiê PLD Intelligence — {profile['name']}",
        "",
        f"**{profile['user_id']}** · {profile['document_type']} {profile['document']} · "
        f"conta {profile.get('account_number') or 'não informada'} · "
        f"score {score} ({LEVEL_LABEL[level]})",
        "",
        "## Quem é o cliente",
        identification(profile),
        "",
        "## Conta e fonte externa",
        (
            f"Conta {profile.get('account_number') or 'não informada'}, situação {status_label(profile.get('account_status'))}, "
            f"saldo ao fim da base {brl(profile.get('balance'))}. "
            f"KYC externo: {status_label(profile.get('kyc_status'))}"
            + (
                f", motivo {reason_label(profile.get('kyc_reason'))}."
                if profile.get("kyc_reason")
                else ", sem motivo adverso registrado."
            )
        ),
        "",
        "## Capacidade financeira no período",
        (
            f"Créditos efetivados: {brl(metrics['credit_sum'])}. Débitos efetivados: {brl(metrics['debit_sum'])}. "
            f"Operações efetivadas: {metrics['tx_count']}. "
            f"Maior volume de débitos em 7 dias: {brl(metrics['volume_7d'])}, "
            f"equivalente a {ratio_label(metrics['ratio_7d'])} vez(es) a renda mensal. "
            f"Período com movimento: {metrics['first_tx']} a {metrics['last_tx']}."
        ),
        "",
        "## Indicadores",
    ]
    if not case["findings"]:
        lines.append("Nenhum indicador do catálogo foi acionado para este cliente.")
    for finding in case["findings"]:
        meta = INDICATORS[finding["code"]]
        lines += [
            f"### {meta['title']} · peso {meta['weight']}",
            finding["narrative"],
            f"Regra aplicada: {meta['rule']}",
            f"Referência: {meta['carta']}",
            "",
        ]
    lines += [
        "## Score",
        (
            f"Score {score} de 100, obtido pela soma dos pesos ({weights}) e limitado a 100. "
            "Faixas: médio a partir de 20, alto a partir de 36, crítico a partir de 60. "
            "O status interno da conta, inclusive BLOCKED_BY_RISK, não entra nessa conta."
        ),
        "",
        "## Contexto operacional",
        _operational_context(profile, metrics),
        "",
        "## Recomendação",
        recommendation(case),
        "",
        "## Fontes consultadas",
        (
            "Cadastro interno de clientes, contas, movimentações de PIX, TED, cartão e saque, "
            "e resultado de KYC externo (bureau). Não houve consulta a notícias, processos ou "
            "listas públicas além do motivo já registrado no KYC sintético."
        ),
        "",
        "## Limite deste texto",
        (
            "Relatório gerado automaticamente por regras explícitas para apoiar o analista. "
            "Não acusa o cliente, não substitui a decisão humana e não constitui comunicação ao COAF. "
            "Base sintética de demonstração, sem clientes reais."
        ),
    ]
    return "\n".join(lines).strip() + "\n"


def _operational_context(profile: dict, metrics: dict) -> str:
    reason = profile.get("reason")
    parts = []
    if reason:
        parts.append(
            f"Situação cadastral interna: {status_label(profile.get('status'))}, motivo {reason_label(reason)}. "
            "Esse campo é exibido como contexto e foi excluído do cálculo dos indicadores, "
            "para a ferramenta reconstituir os sinais a partir da movimentação, do cadastro e do KYC, "
            "sem copiar um bloqueio que já tenha sido decidido."
        )
    else:
        parts.append(f"Situação cadastral interna: {status_label(profile.get('status'))}, sem motivo de restrição.")
    attempts = metrics.get("post_block_attempts") or 0
    if profile.get("status") == "blocked" and attempts:
        parts.append(
            f"Depois da restrição, a base registra {attempts} tentativa(s) recusada(s) com código 04. "
            "Esse fato não soma pontos no score."
        )
    return " ".join(parts)
