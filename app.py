"""PLD Intelligence — fila, dossiê e copiloto para o analista de PLD.

Uso, na pasta do projeto:
  .venv\\Scripts\\python.exe -m streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pld.agent import answer, search_clients  # noqa: E402
from pld.engine import load_analysis, suggest_case  # noqa: E402
from pld.format import brl, ratio_label, status_label  # noqa: E402
from pld.indicators import INDICATORS, LEVEL_LABEL  # noqa: E402
from pld.report import identification, reason_label, render_report  # noqa: E402
from pld.store import (  # noqa: E402
    DECISION_LABEL,
    add_feedback,
    clear_demo_state,
    read_decisions,
    read_feedback,
    write_decision,
)

st.set_page_config(page_title="PLD Intelligence", page_icon="◆", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 4rem; max-width: 1240px;}
    .hero-kicker {color: #F5C518; letter-spacing: .14em; font-size: .75rem; font-weight: 700;}
    .hero-title {font-size: 2rem; line-height: 1.1; margin: .15rem 0 .2rem;}
    .hero-sub {color: #C8C2B4; margin-bottom: .6rem;}
    .pill {display: inline-block; border: 1px solid #3A3422; background: #241F14; color: #F5C518;
           border-radius: 999px; padding: .15rem .55rem; font-size: .78rem; margin-right: .35rem;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Lendo a base sintética e montando a fila…")
def get_store():
    return load_analysis()


def _init() -> None:
    st.session_state.setdefault("user_id", None)
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("page", "fila")
    st.session_state.setdefault("use_llm", False)


def _open(user_id: str) -> None:
    st.session_state.user_id = user_id
    st.session_state.page = "caso"


def _ask(question: str, result: dict) -> None:
    st.session_state.chat.append({"role": "user", "content": question})
    st.session_state.chat.append(
        {
            "role": "assistant",
            "content": answer(question, result, st.session_state.user_id, st.session_state.use_llm),
        }
    )


def _queue_frame(result: dict, decisions: dict, level: str, indicator: str, scope: str) -> pd.DataFrame:
    rows = []
    for user_id in result["ranking"]:
        case = result["cases"][user_id]
        if level != "todos" and case["level"] != level:
            continue
        codes = [finding["code"] for finding in case["findings"]]
        if indicator != "todos" and indicator not in codes:
            continue
        decision = decisions.get(user_id, {}).get("decision", "")
        if scope == "fila" and decision in {"comunicar", "encerrar", "falso_positivo"}:
            continue
        if scope == "decididos" and not decision:
            continue
        profile = case["profile"]
        rows.append(
            {
                "user_id": user_id,
                "Nível": LEVEL_LABEL[case["level"]],
                "Score": case["score"],
                "Cliente": profile["name"],
                "Documento": f"{profile['document_type']} {profile['document']}",
                "Praça": f"{profile['city']}/{profile['state']}",
                "Indicadores": ", ".join(finding["short"] for finding in case["findings"]),
                "Débito 7d / renda": ratio_label(case["metrics"]["ratio_7d"]),
                "Conta": profile.get("account_status") or "",
                "Decisão": DECISION_LABEL.get(decision, "Pendente"),
            }
        )
    return pd.DataFrame(rows)


def _render_header(stats: dict) -> None:
    st.markdown(
        f"""
        <div class="hero-kicker">BANCO DO BRASIL · PROTÓTIPO ACADÊMICO · {stats['period'].upper()}</div>
        <div class="hero-title">PLD Intelligence</div>
        <div class="hero-sub">Evidências claras. Análise mais rápida. Decisão humana.</div>
        """,
        unsafe_allow_html=True,
    )


def _render_kpis(stats: dict, decisions: dict) -> None:
    done = sum(1 for row in decisions.values() if row.get("decision") in {"comunicar", "encerrar", "falso_positivo"})
    cols = st.columns(5)
    values = (
        ("Clientes", f"{stats['n_users']:,}".replace(",", ".")),
        ("Transações", f"{stats['n_transactions']:,}".replace(",", ".")),
        ("Alertas", str(stats["n_alerts"])),
        ("Críticos", str(stats["by_level"].get("critico", 0))),
        ("Já decididos", str(done)),
    )
    for col, (label, value) in zip(cols, values):
        col.metric(label, value)


def _render_findings(case: dict) -> None:
    if not case["findings"]:
        st.info("Nenhum indicador do catálogo foi acionado. O extrato continua disponível para leitura.")
        return
    for finding in case["findings"]:
        with st.container(border=True):
            st.markdown(f"**{finding['title']}**  \npeso {finding['weight']} · `{finding['code']}`")
            st.write(finding["summary"])
            if finding["examples"]:
                table = pd.DataFrame(finding["examples"]).rename(
                columns={"when": "Quando", "type": "Tipo", "direction": "Direção", "amount": "Valor", "detail": "Detalhe"}
            )
            if "Valor" in table:
                table["Valor"] = table["Valor"].map(brl)
            if "Tipo" in table:
                table["Tipo"] = table["Tipo"].replace({"pix": "PIX", "ted": "TED", "card": "Cartão", "cash_withdrawal": "Saque"})
            if "Direção" in table:
                table["Direção"] = table["Direção"].replace({"credit": "crédito", "debit": "débito"})
            st.dataframe(table, hide_index=True, width="stretch")


def _render_movements(frames: dict, case: dict) -> None:
    group = frames["groups"].get(case["user_id"])
    if group is None or group.empty:
        st.caption("Cliente sem movimentação efetivada no período.")
        return
    effective = group[group["status"].isin(["complete", "approved"])].copy()
    if not effective.empty:
        effective["dia"] = effective["created_at"].dt.floor("D")
        daily = (
            effective.pivot_table(index="dia", columns="transaction_direction", values="amount", aggfunc="sum")
            .fillna(0)
            .rename(columns={"credit": "Créditos", "debit": "Débitos"})
        )
        st.line_chart(daily, height=240)
    view = group.sort_values("created_at", ascending=False).head(30).copy()
    view["Quando"] = view["created_at"].dt.strftime("%d/%m/%Y %H:%M")
    view["Valor"] = view["amount"].map(lambda value: f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    st.dataframe(
        view[["Quando", "transaction_type", "transaction_direction", "Valor", "status", "counterparty_account_number"]],
        hide_index=True,
        width="stretch",
    )


def _decide(case: dict, decision: str) -> None:
    codes = [finding["code"] for finding in case["findings"]]
    write_decision(case["user_id"], decision, case["score"], ",".join(codes))
    if decision == "falso_positivo":
        add_feedback(case["user_id"], codes)
    st.session_state.page = "fila"
    st.session_state.user_id = None


def _render_case(frames: dict, result: dict, case: dict, decisions: dict) -> None:
    profile = case["profile"]
    metrics = case["metrics"]
    decision = decisions.get(case["user_id"], {}).get("decision", "")
    if st.button("← Voltar para a fila", key="back"):
        st.session_state.page = "fila"
        st.session_state.user_id = None
        st.rerun()

    st.markdown(f"### {profile['name']}")
    st.markdown(
        f"<span class='pill'>{LEVEL_LABEL[case['level']]} · {case['score']}</span>"
        f"<span class='pill'>{profile['user_id']}</span>"
        f"<span class='pill'>{profile['document_type']} {profile['document']}</span>"
        f"<span class='pill'>{DECISION_LABEL.get(decision, 'Pendente')}</span>",
        unsafe_allow_html=True,
    )
    st.write(identification(profile))

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"**Débito em 7 dias**  \n{brl(metrics['volume_7d'])}")
    c2.markdown(f"**Vezes a renda**  \n{ratio_label(metrics['ratio_7d'])}")
    c3.markdown(f"**Saldo**  \n{brl(profile['balance'])}")
    c4.markdown(f"**KYC externo**  \n{status_label(profile.get('kyc_status'))}")
    st.caption(
        f"Conta {profile.get('account_number') or 'não informada'} · {status_label(profile.get('account_status'))} · "
        f"cadastro interno {status_label(profile.get('status'))}"
        + (f" ({reason_label(profile.get('reason'))})" if profile.get("reason") else "")
        + ". O status interno não entra no score."
    )

    tabs = st.tabs(["Indicadores", "Movimentação", "Relatório"])
    with tabs[0]:
        _render_findings(case)
    with tabs[1]:
        _render_movements(frames, case)
    with tabs[2]:
        report = render_report(case)
        st.markdown(report)
        st.download_button(
            "Baixar dossiê (.md)",
            data=report.encode("utf-8"),
            file_name=f"dossie_{case['user_id']}.md",
            mime="text/markdown",
        )

    st.subheader("Decisão do analista")
    st.caption("A ferramenta não comunica ao COAF e não encerra a conta. Estes botões só registram a decisão desta demonstração.")
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Em análise", width="stretch"):
        _decide(case, "em_analise")
        st.rerun()
    if b2.button("Encaminhar para comunicação", width="stretch"):
        _decide(case, "comunicar")
        st.rerun()
    if b3.button("Encerrar sem comunicação", width="stretch"):
        _decide(case, "encerrar")
        st.rerun()
    if b4.button("Falso positivo", width="stretch"):
        _decide(case, "falso_positivo")
        st.rerun()


def _render_queue(result: dict, decisions: dict, level: str, indicator: str, scope: str) -> None:
    _render_kpis(result["stats"], decisions)
    suggested = suggest_case(result)
    if suggested:
        titles = ", ".join(finding["short"] for finding in suggested["findings"])
        left, right = st.columns([4, 1])
        left.info(
            f"Caso sugerido para a demonstração: **{suggested['profile']['name']}** "
            f"({suggested['user_id']}), score {suggested['score']}. {titles}."
        )
        if right.button("Abrir caso", width="stretch"):
            _open(suggested["user_id"])
            st.rerun()

    view = _queue_frame(result, decisions, level, indicator, scope)
    st.caption(f"{len(view)} casos neste recorte. Clique uma linha para abrir o dossiê.")
    if view.empty:
        st.warning("Nenhum caso neste filtro.")
        return
    token = st.session_state.setdefault("queue_token", 0)
    event = st.dataframe(
        view.drop(columns=["user_id"]),
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key=f"queue-{token}",
    )
    selected = event.selection.rows if event is not None else []
    if selected:
        st.session_state.queue_token = token + 1
        _open(view.iloc[selected[0]]["user_id"])
        st.rerun()

    if result["hubs"]:
        st.subheader("Destinos externos usados por vários clientes")
        hub_frame = pd.DataFrame(
            [
                {
                    "Conta": hub["account"],
                    "Clientes": hub["n_users"],
                    "Envios": hub["n_tx"],
                    "Total": brl(hub["total"]),
                    "Exemplos": ", ".join(hub["sample"]),
                }
                for hub in result["hubs"]
            ]
        )
        st.dataframe(hub_frame, hide_index=True, width="stretch")


def _render_method(result: dict) -> None:
    st.subheader("Como o score é calculado")
    st.write(
        "Cada indicador tem um peso fixo, publicado abaixo. O score do cliente é a soma desses pesos, "
        "limitada a 100. Médio começa em 20, alto em 36 e crítico em 60. "
        "O motivo interno BLOCKED_BY_RISK não é lido por nenhuma regra: a tela mostra a situação da conta "
        "só como contexto, para o protótipo não copiar um bloqueio que o banco já tenha decidido."
    )
    rows = [
        {
            "Indicador": meta["title"],
            "Peso": meta["weight"],
            "Regra": meta["rule"],
            "Tipologia": meta["carta"],
        }
        for meta in INDICATORS.values()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    feedback = read_feedback()
    st.write(
        f"Marcações de falso positivo nesta demonstração: {len(feedback)}. "
        "Elas ficam registradas para a equipe de PLD revisar a regra; não retreinam um modelo sozinhas "
        "e não apagam o caso, que continua consultável no filtro de decididos."
    )
    st.write(
        "Fontes deste protótipo: cadastro, conta, movimentações e KYC externo sintéticos. "
        "Notícias, processos e listas públicas reais ficam como conector futuro. "
        "A redação padrão é determinística. Se houver chave de modelo, o copiloto pode reescrever "
        "a resposta sem receber a base inteira."
    )
    counts = result["stats"]["by_indicator"]
    st.bar_chart(pd.Series({INDICATORS[code]["short"]: count for code, count in counts.items() if count}))


def _render_copilot(result: dict) -> None:
    st.subheader("Copiloto")
    prompts = [
        ("Críticos", "Quantos alertas críticos existem?"),
        ("Explicar o caso", "Por que este cliente foi priorizado?"),
        ("Destinos repetidos", "Quais contas externas recebem de vários clientes?"),
        ("Gerar relatório", "Gere o relatório deste cliente."),
    ]
    cols = st.columns(len(prompts))
    for col, (label, prompt) in zip(cols, prompts):
        if col.button(label, key=f"prompt-{label}", width="stretch"):
            _ask(prompt, result)
            st.rerun()
    for message in st.session_state.chat[-8:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    with st.form("copiloto", clear_on_submit=True):
        typed = st.text_input("Pergunta", placeholder="Pergunte sobre o caso aberto ou sobre a fila", label_visibility="collapsed")
        send = st.form_submit_button("Perguntar")
    if send and typed.strip():
        _ask(typed.strip(), result)
        st.rerun()


def main() -> None:
    _init()
    frames, result = get_store()
    decisions = read_decisions()

    with st.sidebar:
        st.markdown("**PLD Intelligence**")
        st.caption("Apoio ao analista. Não decide e não comunica.")
        query = st.text_input("Buscar cliente", placeholder="Nome, CPF/CNPJ ou U00001")
        if st.button("Buscar", width="stretch") and query.strip():
            matches = search_clients(query, result["cases"])
            if len(matches) == 1:
                _open(matches[0]["user_id"])
                st.rerun()
            elif matches:
                st.session_state["matches"] = [case["user_id"] for case in matches]
            else:
                st.warning("Nenhum cliente encontrado.")
        match_ids = st.session_state.get("matches") or []
        if match_ids:
            labels = {uid: f"{result['cases'][uid]['profile']['name']} · {uid}" for uid in match_ids}
            picked = st.selectbox("Resultados", options=match_ids, format_func=lambda uid: labels[uid])
            if st.button("Abrir resultado", width="stretch"):
                _open(picked)
                st.rerun()
        st.divider()
        level = st.selectbox(
            "Nível",
            options=["todos", "critico", "alto", "medio"],
            format_func=lambda value: "Todos os níveis" if value == "todos" else LEVEL_LABEL[value],
        )
        indicator = st.selectbox(
            "Indicador",
            options=["todos", *INDICATORS.keys()],
            format_func=lambda value: "Todos os indicadores" if value == "todos" else INDICATORS[value]["short"],
        )
        scope = st.selectbox(
            "Fila",
            options=["fila", "todos", "decididos"],
            format_func=lambda value: {"fila": "Ainda na fila", "todos": "Todos os alertas", "decididos": "Já decididos"}[value],
        )
        st.session_state.use_llm = st.toggle(
            "Reescrever com LLM",
            value=st.session_state.use_llm,
            help="Usa OPENAI_API_KEY, GROQ_API_KEY ou PLD_LLM_API_KEY. Sem chave, a resposta continua sendo a das regras.",
        )
        st.divider()
        if st.button("Como o score funciona", width="stretch"):
            st.session_state.page = "metodo"
            st.rerun()
        if st.button("Voltar à fila", width="stretch"):
            st.session_state.page = "fila"
            st.session_state.user_id = None
            st.rerun()
        if st.button("Restaurar demonstração", width="stretch"):
            clear_demo_state()
            st.session_state.chat = []
            st.rerun()

    _render_header(result["stats"])
    page = st.session_state.page
    if page == "metodo":
        _render_method(result)
    elif page == "caso" and st.session_state.user_id in result["cases"]:
        _render_case(frames, result, result["cases"][st.session_state.user_id], decisions)
    else:
        _render_queue(result, decisions, level, indicator, scope)
    _render_copilot(result)


if __name__ == "__main__":
    main()
else:
    main()
