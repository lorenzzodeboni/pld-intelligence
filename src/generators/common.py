"""Constantes e utilitários compartilhados pelos geradores de dados sintéticos."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

SEED = 42

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

START = pd.Timestamp("2026-01-01 00:00:00")
END = pd.Timestamp("2026-04-01 00:00:00")

MINUTE = 60
HOUR = 60 * MINUTE
DAY = 24 * HOUR
END_S = int((END - START).total_seconds())

N_USERS = 5_000
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
CSV_OPTIONS = {"index": False, "sep": ",", "encoding": "utf-8-sig", "float_format": "%.2f"}

USER_COLUMNS = [
    "user_id", "document_type", "document_number", "name", "company_name", "owner_cpf",
    "created_at", "updated_at", "status", "reason", "birth_date", "address", "city",
    "state", "country", "occupation", "income", "net_worth",
]
ACCOUNT_COLUMNS = ["account_id", "account_number", "user_id", "balance", "created_at", "updated_at", "status"]
TRANSACTION_COLUMNS = [
    "transaction_id", "user_id", "counterparty_user_id", "counterparty_account_number",
    "created_at", "updated_at", "transaction_type", "transaction_direction", "amount",
    "status", "status_code", "reason",
]
KYC_COLUMNS = ["user_id", "created_at", "updated_at", "status", "reason"]

# Intensidade relativa de transações cotidianas por hora (0h..23h).
TX_HOUR_WEIGHTS = np.array([
    0.6, 0.35, 0.2, 0.15, 0.15, 0.3, 0.9, 2.0, 3.5, 4.5, 5.0, 5.2,
    5.8, 5.2, 4.8, 4.8, 5.0, 5.3, 5.8, 6.0, 5.5, 4.5, 3.0, 1.6,
])
TX_HOUR_P = TX_HOUR_WEIGHTS / TX_HOUR_WEIGHTS.sum()

INTERNAL_AGENCIES = ("0001", "0147", "0290", "1203", "3051")


def make_generators(seed: int = SEED):
    Faker.seed(seed)
    return np.random.default_rng(seed), random.Random(seed), Faker("pt_BR")


def to_datetime_str(seconds) -> pd.Series:
    offsets = pd.to_timedelta(pd.Series(seconds, dtype="float64").to_numpy(), unit="s")
    return pd.Series(START + offsets).dt.strftime(DATETIME_FORMAT).to_numpy()


def to_money(cents) -> np.ndarray:
    return np.asarray(cents, dtype="float64") / 100.0


def to_cents(value: float) -> int:
    return int(round(float(value) * 100))


def _mod11_digit(digits: str) -> str:
    weights = [2, 3, 4, 5, 6, 7, 8, 9]
    total = sum(int(d) * weights[k % 8] for k, d in enumerate(reversed(digits)))
    digit = 11 - total % 11
    return "0" if digit >= 10 else str(digit)


class AccountNumberFactory:
    """Gera números de conta únicos (formato AAAA-NNNNNNNN-D) para contas internas e externas."""

    def __init__(self, rnd: random.Random):
        self._rnd = rnd
        self._used: set[str] = set()

    def _make(self, agency: str) -> str:
        while True:
            number = f"{self._rnd.randint(0, 99_999_999):08d}"
            account = f"{agency}-{number}-{_mod11_digit(agency + number)}"
            if account not in self._used:
                self._used.add(account)
                return account

    def internal(self) -> str:
        return self._make(self._rnd.choice(INTERNAL_AGENCIES))

    def external(self) -> str:
        while True:
            agency = f"{self._rnd.randint(1, 9999):04d}"
            if agency not in INTERNAL_AGENCIES:
                return self._make(agency)
