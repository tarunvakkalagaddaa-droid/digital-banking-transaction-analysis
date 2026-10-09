"""
Digital Banking Transaction Analysis
------------------------------------
A simple base project that:
  1. Loads transactions from a CSV (or generates sample data if none exists)
  2. Cleans the data
  3. Analyses spending, channels, failures, customers and time trends
  4. Flags suspicious (anomalous) transactions
  5. Saves charts and a text report to the 'output' folder

Run:  python banking_analysis.py
      python banking_analysis.py --file my_transactions.csv

Expected CSV columns:
  transaction_id, customer_id, date, amount, txn_type, channel, category, status
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # no GUI needed; just save images
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = "output"
DATA_FILE = "transactions.csv"


# ----------------------------------------------------------------------
# 1. DATA LOADING / GENERATION
# ----------------------------------------------------------------------
def generate_sample_data(n=5000, seed=42):
    """Create a realistic-looking fake dataset of digital banking transactions."""
    rng = np.random.default_rng(seed)

    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 24 * 60, n), unit="m"
    )
    channels = rng.choice(
        ["Mobile App", "Net Banking", "UPI", "Card", "ATM"],
        n, p=[0.30, 0.15, 0.35, 0.15, 0.05],
    )
    categories = rng.choice(
        ["Groceries", "Shopping", "Bills", "Travel", "Food", "Entertainment",
         "Transfer", "Health"],
        n, p=[0.18, 0.15, 0.15, 0.08, 0.15, 0.07, 0.15, 0.07],
    )
    txn_type = rng.choice(["Debit", "Credit"], n, p=[0.75, 0.25])
    amount = np.round(rng.lognormal(mean=6.5, sigma=1.0, size=n), 2)
    status = rng.choice(["Success", "Failed", "Pending"], n, p=[0.93, 0.05, 0.02])

    df = pd.DataFrame({
        "transaction_id": [f"TXN{100000 + i}" for i in range(n)],
        "customer_id": [f"CUST{rng.integers(1, 301):04d}" for _ in range(n)],
        "date": dates,
        "amount": amount,
        "txn_type": txn_type,
        "channel": channels,
        "category": categories,
        "status": status,
    })

    # Inject a few unusually large transactions so anomaly detection has work to do
    idx = rng.choice(n, 15, replace=False)
    df.loc[idx, "amount"] = np.round(rng.uniform(150000, 500000, 15), 2)

    # Inject some dirty data to demonstrate cleaning
    df.loc[rng.choice(n, 20, replace=False), "amount"] = np.nan
    df = pd.concat([df, df.sample(10, random_state=1)])  # duplicates
    return df.sort_values("date").reset_index(drop=True)


def load_data(path):
    if not os.path.exists(path):
        print(f"[info] '{path}' not found - generating sample data.")
        df = generate_sample_data()
        df.to_csv(path, index=False)
        print(f"[info] Sample data saved to '{path}'.")
        return df
    return pd.read_csv(path)


# ----------------------------------------------------------------------
# 2. CLEANING
# ----------------------------------------------------------------------
def clean_data(df):
    before = len(df)
    df = df.drop_duplicates(subset="transaction_id")
    df = df.dropna(subset=["amount", "date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["amount"] > 0].copy()

    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["weekday"] = df["date"].dt.day_name()
    df["hour"] = df["date"].dt.hour
    print(f"[info] Cleaning: {before} rows -> {len(df)} rows.")
    return df


# ----------------------------------------------------------------------
# 3. ANALYSIS
# ----------------------------------------------------------------------
def overall_summary(df):
    ok = df[df["status"] == "Success"]
    return {
        "Total transactions": len(df),
        "Unique customers": df["customer_id"].nunique(),
        "Total debit amount": ok.loc[ok.txn_type == "Debit", "amount"].sum(),
        "Total credit amount": ok.loc[ok.txn_type == "Credit", "amount"].sum(),
        "Average transaction": df["amount"].mean(),
        "Median transaction": df["amount"].median(),
        "Success rate (%)": (df["status"] == "Success").mean() * 100,
        "Failure rate (%)": (df["status"] == "Failed").mean() * 100,
    }


def monthly_trend(df):
    return df.groupby(["month", "txn_type"])["amount"].sum().unstack(fill_value=0)


def category_spend(df):
    debit = df[(df.txn_type == "Debit") & (df.status == "Success")]
    return debit.groupby("category")["amount"].sum().sort_values(ascending=False)


def channel_usage(df):
    return df.groupby("channel").agg(
        transactions=("transaction_id", "count"),
        total_amount=("amount", "sum"),
        failure_rate=("status", lambda s: (s == "Failed").mean() * 100),
    ).sort_values("transactions", ascending=False)


def top_customers(df, n=10):
    ok = df[df.status == "Success"]
    return (ok.groupby("customer_id")["amount"].agg(["count", "sum"])
              .rename(columns={"count": "transactions", "sum": "total_amount"})
              .sort_values("total_amount", ascending=False).head(n))


def detect_anomalies(df, z_threshold=3):
    """Flag transactions whose amount is far from the customer-wide log-average."""
    log_amt = np.log1p(df["amount"])
    z = (log_amt - log_amt.mean()) / log_amt.std()
    flagged = df[z.abs() > z_threshold].copy()
    flagged["z_score"] = z[flagged.index].round(2)
    return flagged.sort_values("amount", ascending=False)


# ----------------------------------------------------------------------
# 4. CHARTS
# ----------------------------------------------------------------------
def save_charts(df, monthly, cat, chan):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Monthly credit vs debit
    monthly.plot(kind="line", marker="o", figsize=(9, 5))
    plt.title("Monthly Credit vs Debit Amount")
    plt.xlabel("Month"); plt.ylabel("Amount")
    plt.xticks(rotation=45); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/monthly_trend.png", dpi=120); plt.close()

    # Spending by category
    cat.plot(kind="bar", color="steelblue", figsize=(9, 5))
    plt.title("Total Debit Spend by Category")
    plt.ylabel("Amount"); plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/category_spend.png", dpi=120); plt.close()

    # Channel share
    plt.figure(figsize=(6, 6))
    plt.pie(chan["transactions"], labels=chan.index, autopct="%1.1f%%", startangle=90)
    plt.title("Transactions by Channel")
    plt.savefig(f"{OUTPUT_DIR}/channel_share.png", dpi=120); plt.close()

    # Hourly activity
    df.groupby("hour").size().plot(kind="bar", color="seagreen", figsize=(9, 5))
    plt.title("Transactions by Hour of Day")
    plt.xlabel("Hour"); plt.ylabel("Count"); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hourly_activity.png", dpi=120); plt.close()

    # Amount distribution (log scale)
    plt.figure(figsize=(9, 5))
    plt.hist(np.log10(df["amount"]), bins=40, color="darkorange", edgecolor="white")
    plt.title("Distribution of Transaction Amounts (log10)")
    plt.xlabel("log10(amount)"); plt.ylabel("Count"); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/amount_distribution.png", dpi=120); plt.close()


# ----------------------------------------------------------------------
# 5. REPORT
# ----------------------------------------------------------------------
def write_report(summary, cat, chan, top, anomalies):
    lines = ["DIGITAL BANKING TRANSACTION ANALYSIS REPORT", "=" * 45, ""]
    lines.append("OVERALL SUMMARY")
    for k, v in summary.items():
        lines.append(f"  {k:<24}: {v:,.2f}" if isinstance(v, float) else f"  {k:<24}: {v:,}")
    lines += ["", "TOP SPENDING CATEGORIES", cat.round(2).to_string(),
              "", "CHANNEL USAGE", chan.round(2).to_string(),
              "", "TOP 10 CUSTOMERS BY VALUE", top.round(2).to_string(),
              "", f"SUSPICIOUS TRANSACTIONS FLAGGED: {len(anomalies)}",
              anomalies[["transaction_id", "customer_id", "date", "amount",
                         "channel", "z_score"]].head(10).to_string(index=False)]
    text = "\n".join(lines)
    with open(f"{OUTPUT_DIR}/report.txt", "w") as f:
        f.write(text)
    return text


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Digital Banking Transaction Analysis")
    parser.add_argument("--file", default=DATA_FILE, help="Path to transactions CSV")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = clean_data(load_data(args.file))

    summary = overall_summary(df)
    monthly = monthly_trend(df)
    cat = category_spend(df)
    chan = channel_usage(df)
    top = top_customers(df)
    anomalies = detect_anomalies(df)

    save_charts(df, monthly, cat, chan)
    anomalies.to_csv(f"{OUTPUT_DIR}/suspicious_transactions.csv", index=False)
    print(write_report(summary, cat, chan, top, anomalies))
    print(f"\n[done] Charts, report and flagged transactions saved in '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
