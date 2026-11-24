"""Geração de transações normais e anômalas, com saldo simulado em ordem cronológica.

Ordem de construção (evita data leakage):
1. histórico normal de cada conta, gerado sem saber se o usuário terá anomalia ou bloqueio;
2. inserção dos blocos anômalos em usuários sorteados;
3. bloqueio por risco sempre DEPOIS da última transação anômala;
4. remoção das operações posteriores ao bloqueio e inclusão de poucas tentativas recusadas;
5. processamento cronológico único: saldo, recusas por saldo/conta bloqueada e espelho das transferências internas.
"""

import math

import numpy as np
import pandas as pd

from common import DAY, END_S, HOUR, MINUTE, TRANSACTION_COLUMNS, TX_HOUR_P, to_datetime_str, to_money

SCENARIOS = (
    "PIX_BURST",
    "HIGH_VALUE",
    "NIGHT_HIGH_VALUE",
    "FAST_IN_OUT",
    "MANY_SENDERS",
    "CASH_WITHDRAWAL_SEQUENCE",
    "REPEATED_EXTERNAL",
)
ANOMALY_SHARE = 0.04
RISK_BLOCK_SHARE = 0.60
N_MULE_ACCOUNTS = 5

BASE_DAILY_RATE = 0.58

KINDS = ("card", "cash_withdrawal", "pix_ext", "ted_ext", "pix_int", "ted_int")
KIND_P = {
    "CPF": np.array([0.36, 0.115, 0.35, 0.035, 0.125, 0.015]),
    "CNPJ": np.array([0.22, 0.035, 0.42, 0.13, 0.14, 0.055]),
}
EXTERNAL_CREDIT_P = {"CPF": 0.30, "CNPJ": 0.55}

# Fração da renda/faturamento mensal gasta por tipo e sigma da lognormal de valores.
DEBIT_SHARE = {
    "CPF": {"card": 0.30, "cash_withdrawal": 0.08, "pix": 0.30, "ted": 0.15},
    "CNPJ": {"card": 0.10, "cash_withdrawal": 0.02, "pix": 0.38, "ted": 0.35},
}
CREDIT_SHARE = {"CPF": {"pix": 0.15, "ted": 0.05}, "CNPJ": {"pix": 0.70, "ted": 0.30}}
SIGMA = {"card": 0.9, "cash_withdrawal": 0.5, "pix": 1.0, "ted": 0.7}
MIN_AMOUNT = {"card": 2.0, "cash_withdrawal": 20.0, "pix": 1.0, "ted": 1.0}
NORMAL_CASH_LIMIT = {"CPF": 3_000.0, "CNPJ": 5_000.0}

EFFECTIVE = {"complete", "approved", "refund", "reversed"}
BALANCE_CHANGING = {"complete", "approved"}
DECLINE_REASON = {"02": "DECLINED_BY_ISSUER", "03": "DECLINED_BY_RECEIVER", "04": "DECLINED_BY_RISK"}


class _TransactionGenerator:
    def __init__(self, users, accounts, kyc, rng, rnd, factory):
        self.rng, self.rnd, self.factory = rng, rnd, factory
        self.user_ids = users["user_id"].to_numpy()
        self.doc = users["document_type"].to_numpy()
        self.income = users["income"].to_numpy(dtype="float64") / 100
        self.net_worth = users["net_worth"].to_numpy(dtype="float64") / 100
        self.account = accounts["account_number"].to_numpy()
        self.activation = kyc["activation_s"].to_numpy(dtype="float64")
        self.kyc_block = kyc["kyc_block_s"].to_numpy(dtype="float64")
        self.block = np.where(np.isnan(self.kyc_block), math.inf, self.kyc_block)

        n = len(self.user_ids)
        ref_income = np.where(self.doc == "CNPJ", 60_000.0, 3_500.0)
        self.rate = (
            BASE_DAILY_RATE
            * np.clip(np.exp(rng.normal(0.0, 0.45, n)) * (self.income / ref_income) ** 0.3, 0.2, 5.0)
            * np.where(self.doc == "CNPJ", 1.5, 1.0)
        )
        self.budget_factor = rng.uniform(0.75, 1.0, n)
        self.anomaly_ref = np.where(self.doc == "CNPJ", self.income * 0.15, self.income)

        operating = np.where(~np.isnan(self.activation))[0]
        self.pool = operating[np.argsort(self.activation[operating], kind="stable")]
        self.pool_activation = self.activation[self.pool]

        self.events: list[dict] = []
        self.seq = 0
        self.payees: dict[int, list[str]] = {}
        self.payers: dict[int, list[str]] = {}
        self.employer_pool = [factory.external() for _ in range(400)]
        self.mule_accounts = [factory.external() for _ in range(N_MULE_ACCOUNTS)]
        self.anomaly_users: dict[int, str] = {}
        self.anomaly_last_t: dict[int, int] = {}
        self.risk_block: dict[int, int] = {}

    # ------------------------------------------------------------------ utilitários
    def _cents(self, value: float, ttype: str) -> int:
        return int(round(max(float(value), MIN_AMOUNT[ttype]) * 100))

    def _payee(self, i: int) -> str:
        if i not in self.payees:
            self.payees[i] = [self.factory.external() for _ in range(self.rnd.randint(3, 12))]
        return self.rnd.choice(self.payees[i]) if self.rnd.random() < 0.6 else self.factory.external()

    def _payer(self, i: int) -> str:
        if i not in self.payers:
            self.payers[i] = [self.factory.external() for _ in range(self.rnd.randint(2, 6))]
        return self.rnd.choice(self.payers[i]) if self.rnd.random() < 0.5 else self.factory.external()

    def _internal_counterparty(self, i: int, t: int) -> int:
        """Somente contas já ativas no instante t podem ser contraparte interna."""
        k = int(np.searchsorted(self.pool_activation, t, side="left"))
        if k <= 1:
            return -1
        for _ in range(5):
            j = int(self.pool[self.rnd.randrange(k)])
            if j != i:
                return j
        return -1

    def _declined(self, codes, weights):
        code = self.rnd.choices(codes, weights)[0]
        return "declined", code, DECLINE_REASON[code]

    def _failed(self):
        return "failed", "05", self.rnd.choice(["TECHNICAL_ERROR", "PROCESSING_ERROR"])

    def _sample_status(self, ttype: str, direction: str, t: int, anomaly: bool):
        r = self.rnd.random()
        if ttype == "card":
            if r < 0.945 or (anomaly and r >= 0.985):
                return "approved", "00", None
            if r < 0.975:
                return self._declined(("02", "03", "04"), (0.60, 0.15, 0.25))
            if r < 0.985:
                return self._failed()
            return "reversed", "07", "TRANSACTION_REVERSED"
        if ttype == "cash_withdrawal":
            if r < 0.96:
                return "complete", "00", None
            if r < 0.985:
                return self._declined(("02", "04"), (0.75, 0.25))
            return self._failed()
        if direction == "credit":
            if r < 0.985 or (r >= 0.993 and (anomaly or ttype == "ted")):
                return "complete", "00", None
            if r < 0.993:
                return self._failed()
            return "refund", "06", "USER_REQUESTED_REFUND"
        if not anomaly and t >= END_S - 2 * DAY and self.rnd.random() < (0.03 if ttype == "pix" else 0.15):
            return "awaiting", "01", None
        if ttype == "pix":
            if r < 0.955 or (anomaly and r >= 0.988):
                return "complete", "00", None
            if r < 0.975:
                return self._declined(("02", "03", "04"), (0.45, 0.35, 0.20))
            if r < 0.988:
                return self._failed()
            return "refund", "06", "USER_REQUESTED_REFUND"
        if r < 0.965:
            return "complete", "00", None
        if r < 0.985:
            return self._declined(("02", "03", "04"), (0.50, 0.30, 0.20))
        return self._failed()

    def _delay(self, ttype: str, status: str) -> int:
        u = self.rnd.uniform
        if status == "awaiting":
            return 0
        if status in ("complete", "approved"):
            low, high = {"pix": (1, 10), "ted": (10, 30 * MINUTE), "card": (1, 5), "cash_withdrawal": (5, 60)}[ttype]
            return int(u(low, high))
        if status == "declined":
            return int(u(1, 5))
        if status == "failed":
            return int(u(5, 120))
        if status == "refund":
            return int(u(HOUR, 7 * DAY))
        return int(u(30 * MINUTE, 3 * DAY))

    def _event(self, i, t, ttype, direction, cents, cp_user=-1, cp_account=None, scenario=None, status=None):
        t = int(t)
        if status is None:
            status = self._sample_status(ttype, direction, t, scenario is not None)
        if cp_user >= 0:
            cp_account = self.account[cp_user]
        self.events.append({
            "seq": self.seq, "user": i, "t": t, "type": ttype, "dir": direction, "amount": int(cents),
            "cp_user": cp_user, "cp_account": cp_account, "status": status, "scenario": scenario,
        })
        self.seq += 1
        if scenario is not None:
            self.anomaly_last_t[i] = max(self.anomaly_last_t.get(i, -1), t)

    def _transfer_out(self, i, t, ttype, cents, internal_p, scenario, fixed_account=None, fixed_user=-1):
        if fixed_account is not None or fixed_user >= 0:
            j, account = fixed_user, fixed_account
        else:
            j = self._internal_counterparty(i, t) if self.rnd.random() < internal_p else -1
            account = None if j >= 0 else self.factory.external()
        self._event(i, t, ttype, "debit", cents, cp_user=j, cp_account=account, scenario=scenario)

    def _sample_times(self, start: int, end: int, n: int) -> np.ndarray:
        if n <= 0 or end <= start:
            return np.empty(0, dtype="int64")
        chunks, need = [], n
        first_day, last_day = start // DAY, (end - 1) // DAY
        while need > 0:
            m = 2 * need + 8
            t = (
                self.rng.integers(first_day, last_day + 1, m) * DAY
                + self.rng.choice(24, m, p=TX_HOUR_P) * HOUR
                + self.rng.integers(0, HOUR, m)
            )
            t = t[(t > start) & (t < end)][:need]
            chunks.append(t)
            need -= len(t)
        return np.sort(np.concatenate(chunks))

    def _time_of_day(self, t: float) -> int:
        return int(t // DAY * DAY + self.rng.choice(24, p=TX_HOUR_P) * HOUR + self.rng.integers(0, HOUR))

    # ------------------------------------------------------------------ histórico normal
    def _normal_history(self, i: int) -> None:
        start_s = int(self.activation[i])
        end_s = int(min(self.block[i], END_S))
        if end_s - start_s < HOUR:
            return
        doc, income = self.doc[i], self.income[i]

        deposit_t = start_s + int(self.rng.uniform(5 * MINUTE, min(2 * DAY, (end_s - start_s) / 3)))
        deposit = max(self.net_worth[i] * self.rng.uniform(0.03, 0.25), income * self.rng.uniform(0.1, 0.5), 50.0)
        self._event(i, deposit_t, "ted" if self.rnd.random() < 0.6 else "pix", "credit",
                    self._cents(deposit, "ted"), cp_account=self._payer(i))

        if doc == "CPF":
            payday = self.rnd.choice([1, 5, 5, 5, 10, 15, 20, 25, 28])
            salary_type = "ted" if self.rnd.random() < 0.25 else "pix"
            employer = self.rnd.choice(self.employer_pool)
            for month in (1, 2, 3):
                t = (pd.Timestamp(2026, month, payday) - pd.Timestamp(2026, 1, 1)).total_seconds()
                t = int(t + self.rng.uniform(6 * HOUR, 10 * HOUR))
                if deposit_t < t < end_s:
                    self._event(i, t, salary_type, "credit",
                                self._cents(income * self.rng.uniform(0.98, 1.02), salary_type), cp_account=employer)

        start = deposit_t + MINUTE
        days = (end_s - start) / DAY
        times = self._sample_times(start, end_s, int(self.rng.poisson(self.rate[i] * days)))
        n = len(times)
        if n == 0:
            return

        p = KIND_P[doc]
        kinds = self.rng.choice(len(KINDS), n, p=p)
        credit_p = EXTERNAL_CREDIT_P[doc]
        is_credit = np.isin(kinds, (2, 3)) & (self.rng.random(n) < credit_p)

        per_month = self.rate[i] * 30
        expected = {
            ("card", "debit"): per_month * p[0],
            ("cash_withdrawal", "debit"): per_month * p[1],
            ("pix", "debit"): per_month * (p[2] * (1 - credit_p) + p[4]),
            ("ted", "debit"): per_month * (p[3] * (1 - credit_p) + p[5]),
            ("pix", "credit"): per_month * p[2] * credit_p,
            ("ted", "credit"): per_month * p[3] * credit_p,
        }
        night = (times % DAY) < 5 * HOUR

        for k in range(n):
            kind = KINDS[kinds[k]]
            ttype = kind.split("_")[0] if kind.endswith(("_ext", "_int")) else kind
            direction = "credit" if is_credit[k] else "debit"
            share = (CREDIT_SHARE if is_credit[k] else DEBIT_SHARE)[doc][ttype]
            sigma = SIGMA[ttype]
            mean = income * share * self.budget_factor[i] / max(expected[(ttype, direction)], 0.5)
            amount = min(mean * float(np.exp(self.rng.normal(-sigma * sigma / 2, sigma))), 1.5 * income)
            if night[k]:
                amount *= 0.4
            if ttype == "cash_withdrawal":
                amount = min(max(round(amount / 10) * 10, 20.0), NORMAL_CASH_LIMIT[doc])
            cents = self._cents(amount, ttype)
            t = int(times[k])

            if ttype in ("card", "cash_withdrawal"):
                self._event(i, t, ttype, "debit", cents)
            elif direction == "credit":
                self._event(i, t, ttype, "credit", cents, cp_account=self._payer(i))
            elif kind.endswith("_int"):
                j = self._internal_counterparty(i, t)
                self._event(i, t, ttype, "debit", cents, cp_user=j,
                            cp_account=None if j >= 0 else self._payee(i))
            else:
                self._event(i, t, ttype, "debit", cents, cp_account=self._payee(i))

    # ------------------------------------------------------------------ anomalias
    def _pick_anomaly_users(self) -> dict[int, str]:
        total = round(ANOMALY_SHARE * len(self.user_ids))
        counts = {s: total // len(SCENARIOS) + (1 if k < total % len(SCENARIOS) else 0)
                  for k, s in enumerate(SCENARIOS)}
        eligible = [int(i) for i in self.pool
                    if np.isnan(self.kyc_block[i]) and self.activation[i] <= END_S - 10 * DAY]
        cpf = [i for i in eligible if self.doc[i] == "CPF"]
        median_cpf_income = float(np.median(self.income[cpf]))

        chosen: dict[int, str] = {}

        def take(scenario, candidates):
            candidates = [i for i in candidates if i not in chosen]
            for i in self.rng.choice(candidates, counts[scenario], replace=False):
                chosen[int(i)] = scenario

        take("HIGH_VALUE", [i for i in cpf if self.income[i] <= median_cpf_income])
        take("CASH_WITHDRAWAL_SEQUENCE", cpf)
        take("NIGHT_HIGH_VALUE", cpf)
        for scenario in ("PIX_BURST", "FAST_IN_OUT", "MANY_SENDERS", "REPEATED_EXTERNAL"):
            take(scenario, eligible)
        return chosen

    def _anomaly_start(self, i: int) -> int:
        low = self.activation[i] + 2 * DAY
        high = END_S - 8 * DAY
        return self._time_of_day(self.rng.uniform(low, high))

    def _pix_burst(self, i, t0, R, s):
        k = self.rnd.randint(15, 40)
        base = R * self.rng.uniform(0.10, 0.35)
        amounts = np.full(k, round(base)) if self.rnd.random() < 0.5 else base * self.rng.uniform(0.97, 1.03, k)
        times = t0 + np.sort(self.rng.uniform(0, self.rng.uniform(10 * MINUTE, 3 * HOUR), k))
        if self.rnd.random() < 0.6:
            funding = amounts.sum() * self.rng.uniform(1.03, 1.10)
            self._event(i, t0 - self.rng.uniform(2 * HOUR, 24 * HOUR), "ted", "credit",
                        self._cents(funding, "ted"), cp_account=self.factory.external(), scenario=s)
            for t, a in zip(times, amounts):
                self._transfer_out(i, t, "pix", self._cents(a, "pix"), 0.3, s)
        else:
            for t, a in zip(times, amounts):
                self._event(i, t, "pix", "credit", self._cents(a, "pix"),
                            cp_account=self.factory.external(), scenario=s)

    def _high_value(self, i, t0, R, s):
        big = max(R * self.rng.uniform(20, 60), self.net_worth[i] * self.rng.uniform(1.2, 3.0))
        self._event(i, t0, "ted" if self.rnd.random() < 0.7 else "pix", "credit",
                    self._cents(big, "ted"), cp_account=self.factory.external(), scenario=s)
        n_out = self.rnd.randint(1, 2)
        parts = big * self.rng.uniform(0.85, 0.98) * self.rng.dirichlet(np.full(n_out, 3.0))
        t = t0 + self.rng.uniform(3 * HOUR, 30 * HOUR)
        for a in parts:
            self._transfer_out(i, t, "ted" if self.rnd.random() < 0.6 else "pix", self._cents(a, "ted"), 0.0, s)
            t += self.rng.uniform(10 * MINUTE, 5 * HOUR)

    def _night_high_value(self, i, t0, R, s):
        day = t0 // DAY + 1
        k = self.rnd.randint(2, 5)
        amounts = R * self.rng.uniform(3, 12, k)
        funding = amounts.sum() * self.rng.uniform(1.03, 1.10)
        self._event(i, (day - 1) * DAY + self.rng.uniform(10 * HOUR, 20 * HOUR), "ted", "credit",
                    self._cents(funding, "ted"), cp_account=self.factory.external(), scenario=s)
        t = day * DAY + self.rng.uniform(0, 2 * HOUR)
        for a in amounts:
            self._transfer_out(i, t, "pix" if self.rnd.random() < 0.8 else "ted", self._cents(a, "pix"), 0.25, s)
            t += self.rng.uniform(5 * MINUTE, 35 * MINUTE)

    def _fast_in_out(self, i, t0, R, s):
        big = R * self.rng.uniform(8, 40)
        self._event(i, t0, "pix" if self.rnd.random() < 0.6 else "ted", "credit",
                    self._cents(big, "pix"), cp_account=self.factory.external(), scenario=s)
        n_out = self.rnd.randint(1, 3)
        parts = big * self.rng.uniform(0.80, 0.98) * self.rng.dirichlet(np.full(n_out, 3.0))
        t = t0 + self.rng.uniform(5 * MINUTE, 3 * HOUR)
        for a in parts:
            self._transfer_out(i, t, "pix", self._cents(a, "pix"), 0.3, s)
            t += self.rng.uniform(2 * MINUTE, 30 * MINUTE)

    def _many_senders(self, i, t0, R, s):
        k = self.rnd.randint(20, 60)
        times = t0 + np.sort(self.rng.uniform(0, self.rng.uniform(12 * HOUR, 72 * HOUR), k))
        amounts = R * self.rng.uniform(0.05, 0.30, k)
        for t, a in zip(times, amounts):
            self._event(i, t, "pix", "credit", self._cents(a, "pix"),
                        cp_account=self.factory.external(), scenario=s)
        t = times[-1] + self.rng.uniform(30 * MINUTE, 6 * HOUR)
        target_user = self._internal_counterparty(i, int(t)) if self.rnd.random() < 0.2 else -1
        target_account = None if target_user >= 0 else self.factory.external()
        n_out = self.rnd.randint(1, 2)
        parts = amounts.sum() * self.rng.uniform(0.85, 0.95) * self.rng.dirichlet(np.full(n_out, 3.0))
        for a in parts:
            self._transfer_out(i, t, "pix", self._cents(a, "pix"), 0.0, s,
                               fixed_account=target_account, fixed_user=target_user)
            t += self.rng.uniform(2 * MINUTE, 40 * MINUTE)

    def _cash_sequence(self, i, t0, R, s):
        base = round(max(800.0, min(5_000.0, R * self.rng.uniform(0.5, 1.2))) / 50) * 50
        k = self.rnd.randint(5, 15)
        exact = self.rnd.random() < 0.6
        amounts = [base if exact else base + self.rnd.choice([-100, -50, 0, 50, 100]) for _ in range(k)]
        funding = sum(amounts) * self.rng.uniform(1.03, 1.10)
        self._event(i, t0 - self.rng.uniform(1 * HOUR, 24 * HOUR), "pix", "credit",
                    self._cents(funding, "pix"), cp_account=self.factory.external(), scenario=s)
        times = t0 + np.sort(self.rng.uniform(0, self.rng.uniform(1 * HOUR, 6 * HOUR), k))
        for t, a in zip(times, amounts):
            self._event(i, t, "cash_withdrawal", "debit", self._cents(a, "cash_withdrawal"), scenario=s)

    def _repeated_external(self, i, t0, R, s):
        target = self.rnd.choice(self.mule_accounts) if self.rnd.random() < 0.7 else self.factory.external()
        k = self.rnd.randint(5, 12)
        times = t0 + np.sort(self.rng.uniform(0, self.rng.uniform(1 * DAY, 4 * DAY), k))
        amounts = R * self.rng.uniform(0.3, 1.5, k)
        n_in = self.rnd.randint(1, 2)
        funding = amounts.sum() * self.rng.uniform(1.03, 1.10) * self.rng.dirichlet(np.full(n_in, 3.0))
        for a in funding:
            self._event(i, t0 - self.rng.uniform(2 * HOUR, 24 * HOUR), "ted", "credit",
                        self._cents(a, "ted"), cp_account=self.factory.external(), scenario=s)
        for t, a in zip(times, amounts):
            ttype = "pix" if self.rnd.random() < 0.8 else "ted"
            self._transfer_out(i, t, ttype, self._cents(a, ttype), 0.0, s, fixed_account=target)

    def _insert_anomalies(self) -> None:
        builders = {
            "PIX_BURST": self._pix_burst,
            "HIGH_VALUE": self._high_value,
            "NIGHT_HIGH_VALUE": self._night_high_value,
            "FAST_IN_OUT": self._fast_in_out,
            "MANY_SENDERS": self._many_senders,
            "CASH_WITHDRAWAL_SEQUENCE": self._cash_sequence,
            "REPEATED_EXTERNAL": self._repeated_external,
        }
        self.anomaly_users = self._pick_anomaly_users()
        for i, scenario in sorted(self.anomaly_users.items()):
            builders[scenario](i, self._anomaly_start(i), float(self.anomaly_ref[i]), scenario)

    def _apply_risk_blocks(self) -> None:
        """O bloqueio por risco é sempre posterior à última transação anômala do usuário."""
        for i in sorted(self.anomaly_users):
            if self.rnd.random() < RISK_BLOCK_SHARE:
                block_s = int(self.anomaly_last_t[i] + self.rng.uniform(1 * HOUR, 72 * HOUR))
                self.risk_block[i] = block_s
                self.block[i] = block_s
        self.events = [e for e in self.events if e["t"] < self.block[e["user"]]]

    def _post_block_attempts(self) -> None:
        for i in np.where(np.isfinite(self.block))[0]:
            window_start = int(self.block[i]) + 10 * MINUTE
            if window_start > END_S - HOUR:
                continue
            for _ in range(int(self.rng.choice(4, p=[0.30, 0.35, 0.20, 0.15]))):
                t = int(self.rng.uniform(window_start, END_S - MINUTE))
                ttype = self.rnd.choices(["pix", "card", "ted", "cash_withdrawal"], [0.5, 0.3, 0.1, 0.1])[0]
                amount = self.anomaly_ref[i] * float(np.exp(self.rng.normal(np.log(0.05), 0.8)))
                if ttype == "cash_withdrawal":
                    amount = max(round(amount / 10) * 10, 20.0)
                reason = "ACCOUNT_BLOCKED" if self.rnd.random() < 0.6 else "DECLINED_BY_RISK"
                j, account = -1, None
                if ttype in ("pix", "ted"):
                    j = self._internal_counterparty(i, t) if self.rnd.random() < 0.3 else -1
                    account = None if j >= 0 else self._payee(i)
                self._event(i, t, ttype, "debit", self._cents(amount, ttype), cp_user=j, cp_account=account,
                            status=("declined", "04", reason))

    # ------------------------------------------------------------------ processamento cronológico
    def _process(self):
        n = len(self.user_ids)
        balance = [0] * n
        last_move = [-1] * n
        records = []
        for e in sorted(self.events, key=lambda ev: (ev["t"], ev["seq"])):
            i, j, t = e["user"], e["cp_user"], e["t"]
            ttype, direction, amount = e["type"], e["dir"], e["amount"]
            status, code, reason = e["status"]

            if status == "awaiting" and math.isfinite(self.block[i]):
                status, code, reason = "complete", "00", None
            if j >= 0 and self.block[j] <= t and status != "declined":
                status, code, reason = "declined", "03", "ACCOUNT_BLOCKED"
            elif status in EFFECTIVE and direction == "debit" and balance[i] < amount:
                status, code, reason = "declined", "02", "INSUFFICIENT_FUNDS"

            delay = self._delay(ttype, status)
            if status in EFFECTIVE:
                cap = min(END_S, self.block[i], self.block[j] if j >= 0 else math.inf)
                if status in ("refund", "reversed") and t + delay > cap:
                    if cap - t >= 30 * MINUTE:
                        delay = int(self.rnd.uniform(30 * MINUTE, cap - t))
                    else:
                        status = "approved" if ttype == "card" else "complete"
                        code, reason = "00", None
                        delay = self._delay(ttype, status)
                updated = int(max(t, min(t + delay, cap)))
            else:
                updated = min(t + delay, END_S)

            if status in BALANCE_CHANGING:
                if direction == "debit":
                    balance[i] -= amount
                    if j >= 0:
                        balance[j] += amount
                else:
                    balance[i] += amount
            if status in EFFECTIVE:
                last_move[i] = max(last_move[i], updated)
                if j >= 0:
                    last_move[j] = max(last_move[j], updated)

            records.append((i, j, e["cp_account"], t, updated, ttype, direction, amount,
                            status, code, reason, e["scenario"]))
            if j >= 0 and status in ("complete", "refund"):
                records.append((j, i, self.account[i], t, updated, ttype, "credit", amount,
                                status, code, reason, None))

        tx = pd.DataFrame(records, columns=[
            "user_idx", "cp_idx", "counterparty_account_number", "created_s", "updated_s",
            "transaction_type", "transaction_direction", "amount", "status", "status_code",
            "reason", "scenario",
        ])
        tx.insert(0, "transaction_id", [f"T{k + 1:07d}" for k in range(len(tx))])
        tx["user_id"] = self.user_ids[tx["user_idx"].to_numpy()]
        cp = tx["cp_idx"].to_numpy()
        tx["counterparty_user_id"] = np.where(cp >= 0, self.user_ids[np.maximum(cp, 0)], None)
        return tx, balance, last_move

    def run(self):
        for i in self.pool:
            self._normal_history(int(i))
        self._insert_anomalies()
        self._apply_risk_blocks()
        self._post_block_attempts()
        tx, balance, last_move = self._process()
        return tx, {
            "balances": balance,
            "last_move_s": last_move,
            "anomaly_users": {self.user_ids[i]: s for i, s in self.anomaly_users.items()},
            "risk_block": {self.user_ids[i]: b for i, b in self.risk_block.items()},
        }


def generate_transactions(users, accounts, kyc, rng, rnd, factory):
    return _TransactionGenerator(users, accounts, kyc, rng, rnd, factory).run()


def format_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    out = tx.copy()
    out["created_at"] = to_datetime_str(out["created_s"])
    out["updated_at"] = to_datetime_str(out["updated_s"])
    out["amount"] = to_money(out["amount"])
    return out[TRANSACTION_COLUMNS]
