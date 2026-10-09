# Digital Banking Transaction Analysis

Simple Python project that analyses digital banking transactions.

## Setup
    pip install -r requirements.txt

## Run
    python banking_analysis.py                      # uses sample data
    python banking_analysis.py --file your_data.csv # uses your own CSV

CSV columns: transaction_id, customer_id, date, amount, txn_type, channel, category, status

## Output (in the `output/` folder)
- report.txt
- suspicious_transactions.csv
- monthly_trend.png, category_spend.png, channel_share.png, hourly_activity.png, amount_distribution.png
