import pandas as pd
import joblib
import os
from mlxtend.frequent_patterns import apriori, association_rules

CSV_PATH = "data/shop_data.csv"
RULES_PATH = "data/models/association_rules.pkl"
os.makedirs("data/models", exist_ok=True)

def train_rules(min_support=0.005, min_confidence=0.2, top_n_items=100, sample_invoices=None):
    
    df = pd.read_csv(CSV_PATH, encoding='latin-1')
    
    # Optional: sample invoices to reduce size
    if sample_invoices and sample_invoices < df['InvoiceNo'].nunique():
        invoices = df['InvoiceNo'].drop_duplicates().sample(n=sample_invoices, random_state=42)
        df = df[df['InvoiceNo'].isin(invoices)]
    
    # Find top N most frequent items (by number of invoices they appear in)
    item_freq = df.groupby('StockCode')['InvoiceNo'].nunique().sort_values(ascending=False)
    top_items = item_freq.head(top_n_items).index.tolist()
    df = df[df['StockCode'].isin(top_items)]
    
    # Create basket
    basket = df.groupby(['InvoiceNo', 'StockCode'])['Quantity'].sum().unstack().fillna(0)
    # Convert to bool (True/False) for memory efficiency
    basket_bool = basket > 0
    
    print(f"Running apriori with {basket_bool.shape[0]} transactions and {basket_bool.shape[1]} items...")
    freq = apriori(basket_bool, min_support=min_support, use_colnames=True, verbose=1)
    if freq.empty:
        print("No frequent itemsets found. Try lowering min_support.")
        return
    
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    rules['antecedents'] = rules['antecedents'].apply(list)
    rules['consequents'] = rules['consequents'].apply(list)
    joblib.dump(rules, RULES_PATH)
    print(f" Saved {len(rules)} association rules to {RULES_PATH}")

if __name__ == "__main__":
    
    train_rules(min_support=0.005, min_confidence=0.2, top_n_items=100, sample_invoices=5000)