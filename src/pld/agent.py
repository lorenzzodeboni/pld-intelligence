"""Copiloto do analista.

As respostas saem dos dossiês já calculados. Um modelo generativo, se houver
chave configurada, só reescreve esse texto: não recebe a base inteira e não
pode inventar movimentação.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

from .format import brl, ratio_label
from .indicators import INDICATORS, LEVEL_LABEL
from .report import render_report

_ACCENTS = str.maketrans("áàãâäéêíóôõúüç", "aaaaaeeiooouuc")


def search_clients(query: str, cases: dict, limit: int = 8) -> list[dict]:
    text = (query or "").strip()
    if not text:
        return []
    folded = _fold(text)
    digits = re.sub(r"\D", "", text)
    user_id = text.upper()
    ranked: list[tuple[int, dict]] = []
    for case in cases.values():
        profile = case["profile"]
        score = None
        if user_id == profile["user_id"]:
            score = 0
        elif digits and digits == profile["document_digits"]:
            score = 1
        elif folded and folded in _fold(profile["name"]):
            score = 2
        elif profile.get("company_name") and folded in _fold(profile["company_name"]):
            score = 2
        if score is not None:
            ranked.append((score, case))
    ranked.sort(key=lambda item: (item[0], -item[1]["score"]))
    return [case for _, case in ranked[:limit]]


def answer(question: str, result: dict, focus_id: str | None = None, use_llm: bool = False) -> str:
    cases = result["cases"]
    focus = cases.get(focus_id) if focus_id else None
    mentioned = _mentioned_case(question, cases)
    if mentioned:
        focus = mentioned
    draft = _draft(question, result, focus)
    if not use_llm:
        return draft
    rewritten, note = _rewrite(question, draft)
    if rewritten:
        return rewritten + "\n\n_Redação de modelo generativo, restrita às evidências acima._"
    return draft + (f"\n\n_{note}_" if note else "")


def _draft(question: str, result: dict, focus: dict | None) -> str:
    folded = _fold(question)
    stats = result["stats"]

    if any(word in folded for word in ("relatorio", "dossie", "comunicacao ao coaf")):
        if focus is None:
            return "Abra um cliente ou diga o nome, o CPF/CNPJ ou o identificador (U00001) para eu redigir o dossiê."
        return render_report(focus)

    if any(word in folded for word in ("destino", "conta laranja", "concentra", "mesma conta externa", "hub")):
        return _hubs(result)

    if any(word in folded for word in ("quantos", "resumo", "carteira", "fila", "panorama", "base")):
        return _overview(result)

    if any(word in folded for word in ("por que", "porque", "explique", "explicar", "risco deste", "este caso", "esse caso", "selecionad")):
        if focus is None:
            return _overview(result) + "\n\nNenhum cliente está aberto. Busque um nome ou um identificador para eu explicar o caso."
        return _explain(focus)

    code = _indicator_from_question(folded)
    if code:
        return _list_indicator(result, code)

    if any(word in folded for word in ("critico", "prioridade", "prioriz", "quem eu olho", "por onde")):
        return _list_level(result, "critico")

    if "alerta" in folded:
        return _overview(result)

    if focus is not None:
        return _explain(focus)

    return (
        "Consultei o cadastro, as contas, as movimentações e o KYC externo. "
        f"A base tem {stats['n_users']:,} clientes e {stats['n_alerts']:,} alertas no período {stats['period']}.".replace(",", ".")
        + "\n\nPosso explicar o caso aberto, gerar o dossiê, listar os críticos ou mostrar contas externas que recebem de vários clientes. "
        "Exemplos: “por que este cliente foi priorizado?”, “quais são os alertas críticos?”, “quem tem fragmentação de saque?”."
    )


def _explain(case: dict) -> str:
    profile = case["profile"]
    metrics = case["metrics"]
    if not case["findings"]:
        return (
            f"**{profile['name']}** ({profile['user_id']}) não acionou indicador no período. "
            f"Renda declarada {brl(profile['income'])}. Débitos efetivados {brl(metrics['debit_sum'])}. "
            f"Maior semana de débitos: {ratio_label(metrics['ratio_7d'])} vez(es) a renda mensal. "
            "A situação cadastral interna não entra no score."
        )
    lines = [
        f"**{profile['name']}** ({profile['user_id']}) ficou com score **{case['score']}** ({LEVEL_LABEL[case['level']]}).",
        f"Renda declarada {brl(profile['income'])}. Maior semana de débitos: {brl(metrics['volume_7d'])} "
        f"({ratio_label(metrics['ratio_7d'])} vezes a renda).",
        "",
        "O score é a soma dos pesos abaixo. O bloqueio interno, se existir, não entrou nessa conta.",
        "",
    ]
    for finding in case["findings"]:
        lines.append(f"- **{finding['title']}** ({finding['weight']}): {finding['summary']}")
    lines.append("")
    lines.append("Se quiser o texto completo para a análise, peça o relatório deste cliente.")
    return "\n".join(lines)


def _overview(result: dict) -> str:
    stats = result["stats"]
    levels = stats["by_level"]
    parts = [
        f"No período {stats['period']} consultei {stats['n_users']:,} clientes e {stats['n_transactions']:,} transações.".replace(",", "."),
        (
            f"A fila tem **{stats['n_alerts']}** alertas: "
            f"{levels.get('critico', 0)} críticos, {levels.get('alto', 0)} altos e {levels.get('medio', 0)} médios."
        ),
        "",
        "Indicadores acionados:",
    ]
    for code, count in stats["by_indicator"].items():
        if count:
            parts.append(f"- {INDICATORS[code]['title']}: {count}")
    parts.append("")
    parts.append("Os três primeiros da fila:")
    for user_id in result["ranking"][:3]:
        case = result["cases"][user_id]
        titles = ", ".join(finding["short"] for finding in case["findings"])
        parts.append(f"- **{case['profile']['name']}** ({user_id}), score {case['score']}: {titles}")
    return "\n".join(parts)


def _list_level(result: dict, level: str) -> str:
    rows = [result["cases"][uid] for uid in result["ranking"] if result["cases"][uid]["level"] == level]
    if not rows:
        return f"Não há casos no nível {LEVEL_LABEL[level]} com as regras atuais."
    lines = [f"**{len(rows)}** casos {LEVEL_LABEL[level].lower()}s. Os primeiros:"]
    for case in rows[:8]:
        titles = ", ".join(finding["short"] for finding in case["findings"])
        lines.append(f"- **{case['profile']['name']}** ({case['user_id']}), score {case['score']}: {titles}")
    if len(rows) > 8:
        lines.append(f"- … e mais {len(rows) - 8} na fila.")
    return "\n".join(lines)


def _list_indicator(result: dict, code: str) -> str:
    meta = INDICATORS[code]
    rows = [
        result["cases"][uid]
        for uid in result["ranking"]
        if any(finding["code"] == code for finding in result["cases"][uid]["findings"])
    ]
    lines = [f"**{meta['title']}** — {len(rows)} cliente(s).", f"Regra: {meta['rule']}", ""]
    for case in rows[:8]:
        finding = next(item for item in case["findings"] if item["code"] == code)
        lines.append(f"- **{case['profile']['name']}** ({case['user_id']}): {finding['summary']}")
    if len(rows) > 8:
        lines.append(f"- … e mais {len(rows) - 8}.")
    return "\n".join(lines)


def _hubs(result: dict) -> str:
    hubs = result["hubs"]
    if not hubs:
        return "Nenhuma conta externa recebeu PIX ou TED de 3 ou mais clientes nesta base."
    lines = [
        "Contas externas que receberam de vários clientes. Na base sintética, o gerador planta poucos destinos compartilhados; o restante das contas de destino é exclusivo.",
        "",
    ]
    for hub in hubs:
        names = ", ".join(hub["sample"])
        lines.append(
            f"- Conta **{hub['account']}**: {hub['n_users']} clientes, {hub['n_tx']} envios, {brl(hub['total'])}. Exemplos: {names}."
        )
    lines.append("")
    lines.append("Isso não classifica a conta como laranja. Mostra concentração para o analista verificar o beneficiário.")
    return "\n".join(lines)


def _indicator_from_question(folded: str) -> str | None:
    aliases = (
        ("CASH_STRUCTURING", ("saque", "especie", "fragment")),
        ("MULTIPLE_PIX_SHORT_PERIOD", ("rajada", "varios pix", "muitos pix", "pix em sequencia", "curto periodo")),
        ("RAPID_IN_OUT", ("passagem", "entra e sai", "entrada e saida", "in out")),
        ("HIGH_VOLUME_TO_INCOME", ("renda", "incompativ", "capacidade financeira", "volume")),
        ("UNUSUAL_HOUR", ("madruga", "noturn", "horario")),
        ("MANY_ORIGINATORS", ("varias origens", "muitas origens", "varios pagadores", "origens")),
        ("REPEATED_EXTERNAL", ("destino recorrente", "mesma conta", "repetid")),
        ("ADVERSE_KYC", ("sanc", "kyc", "bureau")),
    )
    for code, words in aliases:
        if any(word in folded for word in words):
            return code
    return None


def _mentioned_case(question: str, cases: dict) -> dict | None:
    match = re.search(r"\bU\d{5}\b", question.upper())
    if match and match.group(0) in cases:
        return cases[match.group(0)]
    digits = re.sub(r"\D", "", question)
    if len(digits) in {11, 14}:
        found = search_clients(digits, cases, limit=1)
        if found:
            return found[0]
    folded = _fold(question)
    found = []
    for case in cases.values():
        name = _fold(case["profile"]["name"])
        company = _fold(case["profile"].get("company_name") or "")
        if len(name) > 8 and name in folded:
            found.append(case)
        elif len(company) > 8 and company in folded:
            found.append(case)
    if len(found) == 1:
        return found[0]
    return None


def _fold(text: str) -> str:
    return (text or "").lower().translate(_ACCENTS)


def _rewrite(question: str, draft: str) -> tuple[str | None, str | None]:
    key = os.environ.get("PLD_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not key:
        return None, "LLM desligado na prática: não há OPENAI_API_KEY, GROQ_API_KEY nem PLD_LLM_API_KEY. Mantive a redação das regras."
    if os.environ.get("GROQ_API_KEY") and not os.environ.get("PLD_LLM_BASE_URL") and not os.environ.get("OPENAI_API_KEY"):
        base = "https://api.groq.com/openai/v1"
        model = os.environ.get("PLD_LLM_MODEL", "llama-3.3-70b-versatile")
    else:
        base = os.environ.get("PLD_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.environ.get("PLD_LLM_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Você redige a resposta final de um copiloto de PLD em português. "
                    "Use somente os fatos do rascunho. Não invente transação, contraparte, notícia ou cidade. "
                    "Não acuse o cliente e deixe claro que a decisão é do analista. Se faltar dado, diga que não está na evidência."
                ),
            },
            {"role": "user", "content": f"Pergunta do analista:\n{question}\n\nRascunho com as evidências:\n{draft}"},
        ],
    }
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"].strip()
        return (text or None), None
    except Exception as exc:
        return None, f"O modelo não respondeu ({exc.__class__.__name__}). Mantive a redação das regras."
