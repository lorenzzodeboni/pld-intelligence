"""Gera, valida e salva as quatro bases sintéticas. Uso: python src/generators/generate_all.py"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    CSV_OPTIONS, DATA_DIR, DAY, END_S, N_USERS, START, AccountNumberFactory, make_generators,
)
from generate_accounts import finalize_accounts, format_accounts, generate_accounts  # noqa: E402
from generate_external_kyc import format_external_kyc, generate_external_kyc  # noqa: E402
from generate_transactions import SCENARIOS, format_transactions, generate_transactions  # noqa: E402
from generate_users import age_at, finalize_users, format_users, generate_users  # noqa: E402

ALLOWED_STATUS_CODES = {
    "pix": {("complete", "00"), ("awaiting", "01"), ("declined", "02"), ("declined", "03"),
            ("declined", "04"), ("failed", "05"), ("refund", "06")},
    "card": {("approved", "00"), ("declined", "02"), ("declined", "03"), ("declined", "04"),
             ("failed", "05"), ("reversed", "07")},
    "cash_withdrawal": {("complete", "00"), ("declined", "02"), ("declined", "04"), ("failed", "05")},
    "ted": {("complete", "00"), ("awaiting", "01"), ("declined", "02"), ("declined", "03"),
            ("declined", "04"), ("failed", "05")},
}
REASONS_BY_CODE = {
    "00": {None}, "01": {None},
    "02": {"DECLINED_BY_ISSUER", "INSUFFICIENT_FUNDS"},
    "03": {"DECLINED_BY_RECEIVER", "ACCOUNT_BLOCKED"},
    "04": {"DECLINED_BY_RISK", "ACCOUNT_BLOCKED"},
    "05": {"TECHNICAL_ERROR", "PROCESSING_ERROR"},
    "06": {"USER_REQUESTED_REFUND"},
    "07": {"TRANSACTION_REVERSED"},
}
EFFECTIVE = ["complete", "approved", "refund", "reversed"]
CPF_ONLY_REASONS = {"UNDERAGE", "CPF_IRREGULAR", "CPF_INACTIVE", "DECEASED_HOLDER"}
CNPJ_ONLY_REASONS = {"CNPJ_IRREGULAR", "CNPJ_INACTIVE", "COMPANY_CLOSED", "COMPANY_CANCELLED"}


class Validator:
    def __init__(self):
        self.failures: list[str] = []
        self.count = 0

    def check(self, name: str, condition) -> None:
        self.count += 1
        if not bool(condition):
            self.failures.append(name)


def validate(users, accounts, tx, kyc, info) -> Validator:
    v = Validator()
    n = len(users)
    status = users["status"].to_numpy()
    reason = users["reason"]
    doc = users["document_type"].to_numpy()
    user_index = pd.Series(np.arange(n), index=users["user_id"])

    # Contagens, unicidade e relacionamentos
    v.check("5.000 users", n == N_USERS)
    v.check("5.000 accounts", len(accounts) == N_USERS)
    v.check("5.000 external_kyc", len(kyc) == N_USERS)
    v.check("1 account por user", accounts["user_id"].value_counts().eq(1).all()
            and set(accounts["user_id"]) == set(users["user_id"]))
    v.check("1 KYC por user", kyc["user_id"].is_unique and set(kyc["user_id"]) == set(users["user_id"]))
    v.check("user_id único", users["user_id"].is_unique)
    v.check("account_id único", accounts["account_id"].is_unique)
    v.check("transaction_id único", tx["transaction_id"].is_unique)
    v.check("account_number único", accounts["account_number"].is_unique)
    v.check("document_number único", users["document_number"].is_unique)
    owner = users["owner_cpf"].dropna()
    v.check("owner_cpf único e distinto dos documentos", owner.is_unique and not owner.isin(users["document_number"]).any())
    v.check("tx.user_id válido", tx["user_id"].isin(users["user_id"]).all())
    v.check("tx.counterparty_user_id válido", tx["counterparty_user_id"].dropna().isin(users["user_id"]).all())
    v.check("tabelas alinhadas por user_id", (accounts["user_id"].to_numpy() == users["user_id"].to_numpy()).all()
            and (kyc["user_id"].to_numpy() == users["user_id"].to_numpy()).all())

    # Campos de CPF e CNPJ
    cpf, cnpj = users[doc == "CPF"], users[doc == "CNPJ"]
    v.check("CPF com 11 dígitos", cpf["document_number"].str.fullmatch(r"\d{11}").all())
    v.check("CNPJ com 14 dígitos", cnpj["document_number"].str.fullmatch(r"\d{14}").all())
    v.check("CPF: company_name e owner_cpf NULL", cpf["company_name"].isna().all() and cpf["owner_cpf"].isna().all())
    v.check("CPF: birth_date e occupation preenchidos", cpf["birth_date"].notna().all() and cpf["occupation"].notna().all())
    v.check("CNPJ: company_name e owner_cpf preenchidos", cnpj["company_name"].notna().all()
            and cnpj["owner_cpf"].str.fullmatch(r"\d{11}").all())
    v.check("CNPJ: birth_date NULL", cnpj["birth_date"].isna().all())
    v.check("income e net_worth não negativos", (users["income"] > 0).all() and (users["net_worth"] >= 0).all())

    # Status e reason de users
    v.check("user active -> reason NULL", reason[status == "active"].isna().all())
    v.check("user inactive -> reason preenchida", reason[status == "inactive"].notna().all())
    v.check("user blocked -> reason preenchida", reason[status == "blocked"].notna().all())
    v.check("reason coerente com tipo de documento",
            not reason[doc == "CPF"].isin(CNPJ_ONLY_REASONS).any()
            and not reason[doc == "CNPJ"].isin(CPF_ONLY_REASONS).any())
    underage = np.array([
        doc[k] == "CPF" and age_at(date.fromisoformat(users.at[k, "birth_date"]),
                                   (START + pd.Timedelta(seconds=int(users.at[k, "created_s"]))).date()) < 18
        for k in range(n)
    ])
    v.check("UNDERAGE <-> menor de 18 anos no cadastro", ((reason == "UNDERAGE").to_numpy() == underage).all())

    # KYC x users
    ks, kr = kyc["status"].to_numpy(), kyc["reason"]
    v.check("KYC blocked -> user blocked", (status[ks == "blocked"] == "blocked").all())
    v.check("KYC declined -> user inactive", (status[ks == "declined"] == "inactive").all())
    rejected = ks != "approved"
    v.check("reason do KYC propagada para users", (reason[rejected].to_numpy() == kr[rejected].to_numpy()).all())
    v.check("KYC approved -> reason NULL", kr[ks == "approved"].isna().all())
    approved_blocked = (ks == "approved") & (status == "blocked")
    v.check("KYC approved + user blocked -> BLOCKED_BY_RISK", (reason[approved_blocked] == "BLOCKED_BY_RISK").all())
    v.check("KYC approved -> user active ou BLOCKED_BY_RISK",
            np.isin(status[ks == "approved"], ["active", "blocked"]).all())
    later = kyc["kyc_block_s"].notna().to_numpy()
    v.check("bloqueio posterior do KYC = updated_at do user",
            (kyc["updated_s"].to_numpy()[later] == users["updated_s"].to_numpy()[later]).all())

    # Accounts
    v.check("user blocked/inactive -> account blocked",
            (accounts["status"].to_numpy() == np.where(np.isin(status, ["blocked", "inactive"]), "blocked", "active")).all())

    # Cronologia básica
    for name, df in (("users", users), ("accounts", accounts), ("transactions", tx), ("external_kyc", kyc)):
        v.check(f"{name}: updated_at >= created_at", (df["updated_s"] >= df["created_s"]).all())
        v.check(f"{name}: datas dentro do período", (df["created_s"] >= 0).all() and (df["updated_s"] <= END_S).all())
    v.check("KYC criado junto com o user", (kyc["created_s"].to_numpy() == users["created_s"].to_numpy()).all())
    v.check("account criada após o user", (accounts["created_s"].to_numpy() > users["created_s"].to_numpy()).all())

    # Transações: tipos, direção, valores e códigos
    ttype, direction = tx["transaction_type"], tx["transaction_direction"]
    v.check("card sempre debit", (direction[ttype == "card"] == "debit").all())
    v.check("cash_withdrawal sempre debit", (direction[ttype == "cash_withdrawal"] == "debit").all())
    v.check("amount > 0", (tx["amount"] > 0).all())
    pairs = list(zip(ttype, tx["status"], tx["status_code"]))
    v.check("status/status_code coerentes por tipo", all((s, c) in ALLOWED_STATUS_CODES[t] for t, s, c in pairs))
    v.check("status_code é string", tx["status_code"].map(lambda c: isinstance(c, str)).all())
    v.check("reason coerente com status_code",
            all((None if pd.isna(r) else r) in REASONS_BY_CODE[c] for c, r in zip(tx["status_code"], tx["reason"])))
    v.check("awaiting apenas nas últimas 48h", (tx.loc[tx["status"] == "awaiting", "created_s"] >= END_S - 2 * DAY).all())

    # Contrapartes
    cp = tx["cp_idx"].to_numpy()
    internal = cp >= 0
    account_numbers = accounts["account_number"].to_numpy()
    card_cash = ttype.isin(["card", "cash_withdrawal"]).to_numpy()
    v.check("card/cash sem contraparte", tx.loc[card_cash, "counterparty_user_id"].isna().all()
            and tx.loc[card_cash, "counterparty_account_number"].isna().all())
    v.check("interna: conta corresponde ao counterparty_user_id",
            (tx["counterparty_account_number"].to_numpy()[internal] == account_numbers[cp[internal]]).all())
    external = ~internal & ~card_cash
    v.check("externa: counterparty_user_id NULL e conta fora do banco",
            tx.loc[external, "counterparty_account_number"].notna().all()
            and not tx.loc[external, "counterparty_account_number"].isin(account_numbers).any())
    v.check("sem transferência para si mesmo", (cp[internal] != tx["user_idx"].to_numpy()[internal]).all())
    key = ["created_s", "amount", "transaction_type", "a", "b", "status"]
    sent = tx[internal & (direction == "debit").to_numpy() & tx["status"].isin(["complete", "refund"]).to_numpy()]
    received = tx[internal & (direction == "credit").to_numpy()]
    sent = sent.assign(a=sent["user_idx"], b=sent["cp_idx"])[key].sort_values(key).to_numpy()
    received = received.assign(a=received["cp_idx"], b=received["user_idx"])[key].sort_values(key).to_numpy()
    v.check("transferência interna efetivada possui crédito espelho", sent.shape == received.shape and (sent == received).all())

    # Operação somente após ativação
    activation = kyc["activation_s"].to_numpy()
    user_idx = tx["user_idx"].to_numpy()
    never = np.isnan(activation)
    v.check("users sem KYC aprovado não transacionam", not never[user_idx].any() and not never[cp[internal]].any())
    v.check("transações após a ativação da conta", (tx["created_s"].to_numpy() >= activation[user_idx]).all()
            and (tx["created_s"].to_numpy()[internal] >= activation[cp[internal]]).all())

    # Bloqueios: nada efetivado depois do bloqueio
    block = np.where(status == "blocked", users["updated_s"].to_numpy(dtype="float64"), np.inf)
    created = tx["created_s"].to_numpy()
    own_block = block[user_idx]
    cp_block = np.where(internal, block[np.maximum(cp, 0)], np.inf)
    after_own = created >= own_block
    after_cp = created >= cp_block
    effective = tx["status"].isin(EFFECTIVE).to_numpy()
    v.check("após bloqueio: apenas tentativas recusadas (04)",
            ((tx["status"].to_numpy()[after_own] == "declined") & (tx["status_code"].to_numpy()[after_own] == "04")).all())
    v.check("após bloqueio da contraparte: transferência recusada",
            (tx["status"].to_numpy()[after_cp] == "declined").all())
    v.check("efetivadas concluídas antes do bloqueio",
            (tx["updated_s"].to_numpy()[effective] <= np.minimum(own_block, cp_block)[effective]).all())
    v.check("ACCOUNT_BLOCKED só com conta bloqueada", (after_own | after_cp)[(tx["reason"] == "ACCOUNT_BLOCKED").to_numpy()].all())

    # Saldo
    sign = np.where(direction == "credit", 1, -1) * tx["status"].isin(["complete", "approved"]).to_numpy()
    computed = pd.Series(sign * tx["amount"].to_numpy()).groupby(user_idx).sum().reindex(range(n), fill_value=0)
    v.check("balance = soma das transações efetivadas", (computed.to_numpy() == accounts["balance"].to_numpy()).all())
    v.check("balance não negativo", (accounts["balance"] >= 0).all())
    last_effective = pd.Series(tx["updated_s"].to_numpy()[effective]).groupby(user_idx[effective]).max()
    v.check("account.updated_at >= última movimentação",
            (accounts["updated_s"].to_numpy()[last_effective.index] >= last_effective.to_numpy()).all())

    # Anomalias e bloqueio por risco
    anomaly_users = info["anomaly_users"]
    share = len(anomaly_users) / n
    v.check("3% a 5% dos usuários com anomalias", 0.03 <= share <= 0.05)
    anomaly_tx = tx[tx["scenario"].notna()]
    last_anomaly = anomaly_tx.groupby("user_id")["updated_s"].max()
    risk_users = users.loc[reason == "BLOCKED_BY_RISK", "user_id"]
    v.check("todo BLOCKED_BY_RISK possui anomalia prévia", risk_users.isin(list(anomaly_users)).all()
            and risk_users.isin(last_anomaly.index).all())
    risk_updated = users.set_index("user_id").loc[risk_users, "updated_s"]
    v.check("bloqueio por risco posterior às transações anômalas",
            (risk_updated.to_numpy() > last_anomaly.reindex(risk_users).to_numpy()).all())
    v.check("registro interno de bloqueios = users", set(info["risk_block"]) == set(risk_users))
    v.check("todos os cenários presentes", set(anomaly_users.values()) == set(SCENARIOS))
    v.check("índices de usuário consistentes", (user_index[tx["user_id"]].to_numpy() == user_idx).all())
    return v


def report(users, accounts, tx, kyc, info) -> None:
    def line(label, series):
        counts = series.value_counts()
        print(f"- {label}: " + " / ".join(f"{k} {v:,}" for k, v in counts.items()).replace(",", "."))

    def fmt(value):
        return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    amount = tx["amount"] / 100
    balance = accounts["balance"] / 100
    pix_ted = tx[tx["transaction_type"].isin(["pix", "ted"])]
    internal = pix_ted["counterparty_user_id"].notna()
    risk = users["reason"].eq("BLOCKED_BY_RISK")
    risk_ids = set(users.loc[risk, "user_id"])
    scenario_counts = pd.Series(info["anomaly_users"]).value_counts()

    print("USERS")
    print(f"- total: {len(users):,}".replace(",", "."))
    line("document_type", users["document_type"])
    line("status", users["status"])
    print("\nACCOUNTS")
    print(f"- total: {len(accounts):,}".replace(",", "."))
    line("status", accounts["status"])
    print("\nTRANSACTIONS")
    print(f"- total: {len(tx):,}".replace(",", "."))
    line("transaction_type", tx["transaction_type"])
    line("direction", tx["transaction_direction"])
    print(f"- pix/ted internos: {int(internal.sum()):,} / externos: {int((~internal).sum()):,}".replace(",", "."))
    line("status", tx["status"])
    line("status_code", tx["status_code"].sort_values())
    print("\nKYC")
    line("status", kyc["status"])
    print("\nANOMALIAS")
    print(f"- usuários com anomalias: {len(info['anomaly_users'])} ({len(info['anomaly_users']) / len(users):.1%})")
    for scenario in SCENARIOS:
        blocked = sum(1 for u, s in info["anomaly_users"].items() if s == scenario and u in risk_ids)
        print(f"  - {scenario}: {scenario_counts.get(scenario, 0)} (bloqueados por risco: {blocked})")
    print(f"- posteriormente bloqueados por risco: {int(risk.sum())}")
    print("\nVALORES")
    print(f"- average transaction amount: {fmt(amount.mean())}")
    print(f"- median transaction amount: {fmt(amount.median())}")
    print(f"- total transaction volume: {fmt(amount.sum())}")
    print(f"- average balance: {fmt(balance.mean())}")
    print(f"- median balance: {fmt(balance.median())}")


def main() -> None:
    rng, rnd, fake = make_generators()
    factory = AccountNumberFactory(rnd)

    users = generate_users(rng, rnd, fake)
    kyc = generate_external_kyc(users, rng, rnd)
    accounts = generate_accounts(users, rng, factory)
    tx, info = generate_transactions(users, accounts, kyc, rng, rnd, factory)
    users = finalize_users(users, kyc, info["risk_block"], rng)
    accounts = finalize_accounts(accounts, users, info["balances"], info["last_move_s"])

    result = validate(users, accounts, tx, kyc, info)
    if result.failures:
        print("VALIDAÇÕES FALHARAM:")
        for name in result.failures:
            print(f"- {name}")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    format_users(users).to_csv(DATA_DIR / "users.csv", **CSV_OPTIONS)
    format_accounts(accounts).to_csv(DATA_DIR / "accounts.csv", **CSV_OPTIONS)
    format_transactions(tx).to_csv(DATA_DIR / "transactions.csv", **CSV_OPTIONS)
    format_external_kyc(kyc).to_csv(DATA_DIR / "external_kyc.csv", **CSV_OPTIONS)

    report(users, accounts, tx, kyc, info)
    print(f"\nVALIDAÇÕES: {result.count} verificações OK")


if __name__ == "__main__":
    main()
