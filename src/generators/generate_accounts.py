"""Geração das contas (1 conta por usuário)."""

import numpy as np
import pandas as pd

from common import ACCOUNT_COLUMNS, to_datetime_str, to_money


def generate_accounts(users: pd.DataFrame, rng, factory) -> pd.DataFrame:
    """A conta é aberta segundos após o cadastro e antes do resultado do KYC."""
    n = len(users)
    return pd.DataFrame({
        "account_id": [f"A{k + 1:05d}" for k in range(n)],
        "account_number": [factory.internal() for _ in range(n)],
        "user_id": users["user_id"].to_numpy(),
        "created_s": users["created_s"].to_numpy(dtype="int64") + rng.integers(30, 111, n),
    })


def finalize_accounts(accounts: pd.DataFrame, users: pd.DataFrame, balances, last_move_s) -> pd.DataFrame:
    """Saldo = soma das transações efetivadas; updated_at = última movimentação ou mudança de status."""
    accounts = accounts.copy()
    blocked = users["status"].isin(["blocked", "inactive"]).to_numpy()
    last_move = np.asarray(last_move_s, dtype="int64")
    created = accounts["created_s"].to_numpy(dtype="int64")
    status_change = np.where(blocked, users["updated_s"].to_numpy(dtype="int64"), -1)

    accounts["balance"] = np.asarray(balances, dtype="int64")
    accounts["status"] = np.where(blocked, "blocked", "active")
    accounts["updated_s"] = np.maximum.reduce([created, last_move, status_change])
    return accounts


def format_accounts(accounts: pd.DataFrame) -> pd.DataFrame:
    out = accounts.copy()
    out["created_at"] = to_datetime_str(out["created_s"])
    out["updated_at"] = to_datetime_str(out["updated_s"])
    out["balance"] = to_money(out["balance"])
    return out[ACCOUNT_COLUMNS]
