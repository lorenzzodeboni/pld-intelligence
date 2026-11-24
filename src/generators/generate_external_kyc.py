"""Geração do resultado do KYC externo (bureau) por usuário."""

from datetime import date

import numpy as np
import pandas as pd

from common import DAY, END_S, HOUR, KYC_COLUMNS, MINUTE, START, to_datetime_str
from generate_users import age_at

DECLINED_SHARE = 0.04
BLOCKED_SHARE = 0.03
BLOCKED_AT_ONBOARDING_SHARE = 0.40

DECLINED_REASONS = {
    "CPF": ["CPF_IRREGULAR", "CPF_INACTIVE", "DOCUMENT_NOT_FOUND", "DOCUMENT_DATA_MISMATCH"],
    "CNPJ": [
        "CNPJ_IRREGULAR", "CNPJ_INACTIVE", "DOCUMENT_NOT_FOUND", "DOCUMENT_DATA_MISMATCH",
        "COMPANY_CLOSED", "COMPANY_CANCELLED",
    ],
}
ONBOARDING_BLOCK_REASONS = {
    "CPF": ["SANCTIONS_MATCH", "AML_RISK", "SUSPECTED_FRAUD", "CPF_IRREGULAR"],
    "CNPJ": ["SANCTIONS_MATCH", "AML_RISK", "SUSPECTED_FRAUD", "CNPJ_IRREGULAR"],
}
# Bloqueios que chegam do bureau alguns dias depois da abertura da conta.
LATER_BLOCK_REASONS = {
    "CPF": (["CPF_IRREGULAR", "CPF_INACTIVE", "DECEASED_HOLDER"], [0.70, 0.20, 0.10]),
    "CNPJ": (["CNPJ_IRREGULAR", "CNPJ_INACTIVE", "COMPANY_CLOSED"], [0.70, 0.15, 0.15]),
}


def generate_external_kyc(users: pd.DataFrame, rng, rnd) -> pd.DataFrame:
    """Retorna o KYC com colunas auxiliares:
    activation_s = momento em que a conta pode operar (NaN se nunca operou);
    kyc_block_s  = bloqueio posterior vindo do bureau (NaN se não houver).
    """
    n = len(users)
    doc = users["document_type"].to_numpy()
    created_s = users["created_s"].to_numpy(dtype="int64")

    status = np.full(n, "approved", dtype=object)
    reason = np.full(n, None, dtype=object)
    decision_s = created_s + rng.integers(2 * MINUTE, HOUR, n)
    kyc_block_s = np.full(n, np.nan)

    underage = np.array([
        doc[k] == "CPF" and age_at(date.fromisoformat(users.at[k, "birth_date"]),
                                   (START + pd.Timedelta(seconds=int(created_s[k]))).date()) < 18
        for k in range(n)
    ])
    status[underage] = "declined"
    reason[underage] = "UNDERAGE"

    n_declined_other = max(round(DECLINED_SHARE * n) - int(underage.sum()), 0)
    candidates = np.where(status == "approved")[0]
    declined = rng.choice(candidates, n_declined_other, replace=False)
    status[declined] = "declined"
    for k in declined:
        reason[k] = rnd.choice(DECLINED_REASONS[doc[k]])

    n_blocked = round(BLOCKED_SHARE * n)
    n_onboarding = round(n_blocked * BLOCKED_AT_ONBOARDING_SHARE)
    candidates = np.where(status == "approved")[0]
    onboarding = rng.choice(candidates, n_onboarding, replace=False)
    status[onboarding] = "blocked"
    for k in onboarding:
        reason[k] = rnd.choice(ONBOARDING_BLOCK_REASONS[doc[k]])

    rejected = status != "approved"
    decision_s[rejected] = created_s[rejected] + rng.integers(10 * MINUTE, 36 * HOUR, int(rejected.sum()))

    candidates = np.where((status == "approved") & (decision_s + 12 * DAY <= END_S))[0]
    later = rng.choice(candidates, n_blocked - n_onboarding, replace=False)
    status[later] = "blocked"
    for k in later:
        options, weights = LATER_BLOCK_REASONS[doc[k]]
        reason[k] = rnd.choices(options, weights)[0]
        kyc_block_s[k] = decision_s[k] + int(rng.integers(2 * DAY, 10 * DAY))

    operates = (status == "approved") | ~np.isnan(kyc_block_s)
    updated_s = np.where(np.isnan(kyc_block_s), decision_s, kyc_block_s).astype("int64")

    return pd.DataFrame({
        "user_id": users["user_id"].to_numpy(),
        "created_s": created_s,
        "updated_s": updated_s,
        "status": status,
        "reason": reason,
        "decision_s": decision_s,
        "activation_s": np.where(operates, decision_s, np.nan),
        "kyc_block_s": kyc_block_s,
    })


def format_external_kyc(kyc: pd.DataFrame) -> pd.DataFrame:
    out = kyc.copy()
    out["created_at"] = to_datetime_str(out["created_s"])
    out["updated_at"] = to_datetime_str(out["updated_s"])
    return out[KYC_COLUMNS]
