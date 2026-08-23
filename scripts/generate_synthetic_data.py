"""
Phase A — Synthetic Data Generator for Fraud Network Mapper.

Generates a realistic-looking dataset of accounts + transactions / logins,
with planted fraud rings that share IPs, devices, phone numbers, and bank
accounts. Ground-truth `is_fraud` labels are attached to every account so the
GNN (Phase C) can be trained in a supervised manner.

Outputs (in /data):
  - accounts.csv          one row per account, with is_fraud label
  - logins.csv            one row per login attempt (IP, device, timestamp)
  - transactions.csv      one row per transaction (bank, amount, counterparty)
  - shared_attributes.csv long-form edges listing every shared attribute
  - fraud_rings.json      description of each planted ring (for explain panel)
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
from faker import Faker

# Reproducibility — every demo run produces identical numbers.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
fake = Faker()
Faker.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class GeneratorConfig:
    n_normal_accounts: int = 300
    n_fraud_rings: int = 5
    ring_size_range: tuple = (4, 8)          # accounts per ring
    logins_per_account_range: tuple = (3, 12)
    transactions_per_account_range: tuple = (2, 10)
    start_date: datetime = field(
        default_factory=lambda: datetime(2026, 1, 1)
    )
    end_date: datetime = field(
        default_factory=lambda: datetime(2026, 6, 1)
    )


# ---------------------------------------------------------------------------
# Helper: shared attribute pools
# ---------------------------------------------------------------------------
def make_shared_pool(pool_size: int, kind: str) -> List[str]:
    """Create a pool of `pool_size` fake-but-realistic shared identifiers."""
    pool = set()
    while len(pool) < pool_size:
        if kind == "ip":
            pool.add(fake.ipv4_private())
        elif kind == "device":
            pool.add(f"dev-{uuid.uuid4().hex[:8]}")
        elif kind == "phone":
            pool.add(fake.msisdn()[:13])
        elif kind == "bank":
            pool.add(f"BA-{random.randint(10_000_000, 99_999_999)}")
        else:
            raise ValueError(f"unknown pool kind {kind!r}")
    return list(pool)


# ---------------------------------------------------------------------------
# Account model
# ---------------------------------------------------------------------------
@dataclass
class Account:
    account_id: str
    name: str
    email: str
    age_days: int
    is_fraud: bool = False
    ring_id: str | None = None
    shared_ips: List[str] = field(default_factory=list)
    shared_devices: List[str] = field(default_factory=list)
    shared_phones: List[str] = field(default_factory=list)
    shared_banks: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
def generate(config: GeneratorConfig) -> dict:
    accounts: List[Account] = []
    logins: list[dict] = []
    transactions: list[dict] = []
    shared_edges: list[dict] = []
    rings_meta: list[dict] = []

    # ---------------- Normal accounts ----------------
    for i in range(config.n_normal_accounts):
        acc = Account(
            account_id=f"ACC-{10000 + i}",
            name=fake.name(),
            email=fake.email(),
            age_days=random.randint(30, 2000),
        )
        accounts.append(acc)

    # ---------------- Planted fraud rings ----------------
    # Each ring: 4–8 accounts, all share a tight pool of IPs/devices/phones.
    # Rings vary in "tightness" — tighter rings -> higher fraud signal.
    for ring_idx in range(config.n_fraud_rings):
        ring_id = f"RING-{ring_idx + 1:02d}"
        ring_size = random.randint(*config.ring_size_range)
        tightness = random.choice(["tight", "medium", "loose"])

        # Tightness controls how many shared attributes the ring members have.
        n_ips = {"tight": 3, "medium": 5, "loose": 8}[tightness]
        n_devices = {"tight": 2, "medium": 3, "loose": 4}[tightness]
        n_phones = {"tight": 2, "medium": 3, "loose": 4}[tightness]
        n_banks = {"tight": 1, "medium": 2, "loose": 3}[tightness]

        ring_ips = make_shared_pool(n_ips, "ip")
        ring_devices = make_shared_pool(n_devices, "device")
        ring_phones = make_shared_pool(n_phones, "phone")
        ring_banks = make_shared_pool(n_banks, "bank")

        ring_account_ids: list[str] = []
        for j in range(ring_size):
            acc = Account(
                account_id=f"ACC-F{ring_idx:02d}{j:02d}",
                name=fake.name(),
                email=fake.email(),
                age_days=random.randint(2, 90),  # younger = more suspicious
                is_fraud=True,
                ring_id=ring_id,
                shared_ips=ring_ips.copy(),
                shared_devices=ring_devices.copy(),
                shared_phones=ring_phones.copy(),
                shared_banks=ring_banks.copy(),
            )
            accounts.append(acc)
            ring_account_ids.append(acc.account_id)

        # Record shared-attribute edges between every pair in the ring.
        for i in range(ring_size):
            for k in range(i + 1, ring_size):
                a, b = ring_account_ids[i], ring_account_ids[k]
                for ip in ring_ips:
                    shared_edges.append(
                        {"src": a, "dst": b, "attribute": "ip", "value": ip}
                    )
                for dev in ring_devices:
                    shared_edges.append(
                        {"src": a, "dst": b, "attribute": "device", "value": dev}
                    )
                for ph in ring_phones:
                    shared_edges.append(
                        {"src": a, "dst": b, "attribute": "phone", "value": ph}
                    )
                for bk in ring_banks:
                    shared_edges.append(
                        {"src": a, "dst": b, "attribute": "bank_account", "value": bk}
                    )

        rings_meta.append(
            {
                "ring_id": ring_id,
                "size": ring_size,
                "tightness": tightness,
                "account_ids": ring_account_ids,
                "shared_ips": ring_ips,
                "shared_devices": ring_devices,
                "shared_phones": ring_phones,
                "shared_banks": ring_banks,
            }
        )

    # ---------------- Generate per-account events ----------------
    date_span = (config.end_date - config.start_date).total_seconds()

    for acc in accounts:
        # Logins
        n_logins = random.randint(*config.logins_per_account_range)
        for _ in range(n_logins):
            ts = config.start_date + timedelta(
                seconds=random.randint(0, int(date_span))
            )
            if acc.is_fraud:
                ip = random.choice(acc.shared_ips)
                device = random.choice(acc.shared_devices)
            else:
                ip = fake.ipv4_private()
                device = f"dev-{uuid.uuid4().hex[:8]}"
            logins.append(
                {
                    "login_id": f"LOG-{uuid.uuid4().hex[:10]}",
                    "account_id": acc.account_id,
                    "timestamp": ts.isoformat(timespec="seconds"),
                    "ip_address": ip,
                    "device_id": device,
                    "user_agent": fake.user_agent(),
                    "success": random.random() > 0.05,
                }
            )

        # Transactions
        n_tx = random.randint(*config.transactions_per_account_range)
        for _ in range(n_tx):
            ts = config.start_date + timedelta(
                seconds=random.randint(0, int(date_span))
            )
            if acc.is_fraud:
                bank = random.choice(acc.shared_banks)
                amount = round(
                    random.uniform(500, 25_000), 2
                )  # fraud amounts tend to be larger
            else:
                bank = f"BA-{random.randint(10_000_000, 99_999_999)}"
                amount = round(random.uniform(5, 1500), 2)
            transactions.append(
                {
                    "tx_id": f"TX-{uuid.uuid4().hex[:10]}",
                    "account_id": acc.account_id,
                    "timestamp": ts.isoformat(timespec="seconds"),
                    "bank_account": bank,
                    "amount": amount,
                    "counterparty": fake.company()[:30],
                    "currency": "USD",
                }
            )

    # ---------------- Serialize to dataframes ----------------
    accounts_df = pd.DataFrame(
        [
            {
                "account_id": a.account_id,
                "name": a.name,
                "email": a.email,
                "age_days": a.age_days,
                "is_fraud": int(a.is_fraud),
                "ring_id": a.ring_id or "",
            }
            for a in accounts
        ]
    )

    return {
        "accounts": accounts_df,
        "logins": pd.DataFrame(logins),
        "transactions": pd.DataFrame(transactions),
        "shared_edges": pd.DataFrame(shared_edges),
        "rings": rings_meta,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic fraud-network data."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data"),
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--n-normal",
        type=int,
        default=300,
        help="Number of benign accounts to generate.",
    )
    parser.add_argument(
        "--n-rings",
        type=int,
        default=5,
        help="Number of fraud rings to plant.",
    )
    args = parser.parse_args()

    cfg = GeneratorConfig(
        n_normal_accounts=args.n_normal,
        n_fraud_rings=args.n_rings,
    )

    print(f"Generating synthetic dataset (seed={RANDOM_SEED})…")
    out = generate(cfg)
    args.out.mkdir(parents=True, exist_ok=True)

    accounts_path = args.out / "accounts.csv"
    logins_path = args.out / "logins.csv"
    tx_path = args.out / "transactions.csv"
    edges_path = args.out / "shared_attributes.csv"
    rings_path = args.out / "fraud_rings.json"

    out["accounts"].to_csv(accounts_path, index=False)
    out["logins"].to_csv(logins_path, index=False)
    out["transactions"].to_csv(tx_path, index=False)
    out["shared_edges"].to_csv(edges_path, index=False)
    rings_path.write_text(json.dumps(out["rings"], indent=2))

    # Also write a small, self-contained sample for the frontend demo
    sample_dir = args.out / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    fraud_ids = set(
        out["accounts"].loc[out["accounts"]["is_fraud"] == 1, "account_id"]
    )
    # Pick 80 normals + every fraud account
    normal_ids = set(
        out["accounts"].loc[out["accounts"]["is_fraud"] == 0, "account_id"]
        .sample(n=80, random_state=RANDOM_SEED)
    )
    sample_ids = fraud_ids | normal_ids

    def _filter(df: pd.DataFrame, col: str) -> pd.DataFrame:
        return df[df[col].isin(sample_ids)].copy()

    _filter(out["accounts"], "account_id").to_csv(
        sample_dir / "accounts.csv", index=False
    )
    _filter(out["logins"], "account_id").to_csv(
        sample_dir / "logins.csv", index=False
    )
    _filter(out["transactions"], "account_id").to_csv(
        sample_dir / "transactions.csv", index=False
    )
    edges_df = out["shared_edges"]
    edges_df[
        edges_df["src"].isin(sample_ids) & edges_df["dst"].isin(sample_ids)
    ].to_csv(sample_dir / "shared_attributes.csv", index=False)
    (sample_dir / "fraud_rings.json").write_text(
        json.dumps(out["rings"], indent=2)
    )

    # Summary
    n_fraud = int(out["accounts"]["is_fraud"].sum())
    print(f"  accounts      : {len(out['accounts'])}  ({n_fraud} fraud)")
    print(f"  logins        : {len(out['logins'])}")
    print(f"  transactions  : {len(out['transactions'])}")
    print(f"  shared edges  : {len(out['shared_edges'])}")
    print(f"  rings         : {len(out['rings'])}")
    print(f"\nWrote:")
    print(f"  {accounts_path}")
    print(f"  {logins_path}")
    print(f"  {tx_path}")
    print(f"  {edges_path}")
    print(f"  {rings_path}")
    print(f"\nDemo sample subset ({len(sample_ids)} accounts) -> {sample_dir}/")


if __name__ == "__main__":
    main()
