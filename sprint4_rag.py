import os
import pandas as pd
from dotenv import load_dotenv
import anthropic

# Load API key
load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

print("=" * 50)
print("SPRINT 4 — RAG PIPELINE")
print("=" * 50)

# ─────────────────────────────────────────
# PART 1 — YOUR FIRST CLAUDE CALL
# ─────────────────────────────────────────
print("\n[PART 1] First Claude call...\n")

message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=300,
    system="You are a supply chain analyst at Stanley Black and Decker.",
    messages=[
        {
            "role": "user",
            "content": "In one sentence — what is demand forecasting?"
        }
    ]
)

print("Claude says:")
print(message.content[0].text)
print(f"\nTokens used: {message.usage.input_tokens} input, {message.usage.output_tokens} output")

# ─────────────────────────────────────────
# PART 2 — GIVE CLAUDE YOUR SBD DATA
# ─────────────────────────────────────────
print("\n[PART 2] Giving Claude your actual SBD forecast data...\n")

# Load your forecast data
# First generate it from your sales data
df = pd.read_csv('data/sbd_sales_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['sku_id','warehouse','date']).reset_index(drop=True)

# Get the last 4 weeks of data for each SKU and warehouse
# This simulates your model's forecast output
latest = df.groupby(['sku_id','warehouse']).tail(4)

# Calculate average demand and simple risk score
summary = df.groupby(['sku_id','warehouse']).agg(
    avg_demand=('demand','mean'),
    last_inventory=('inventory','last'),
    total_promos=('is_promo','sum')
).reset_index()

# Simple risk score — weeks of supply remaining
summary['weeks_of_supply'] = (summary['last_inventory'] / 
                               summary['avg_demand']).round(1)
summary['risk'] = summary['weeks_of_supply'].apply(
    lambda x: 'HIGH' if x < 16 else 'MEDIUM' if x < 20 else 'LOW'
)

# Get top 5 HIGH risk SKUs
high_risk = summary[summary['risk']=='HIGH'].sort_values(
    'weeks_of_supply').head(5)


print(summary[['sku_id','warehouse','weeks_of_supply']].sort_values('weeks_of_supply').head(10).to_string(index=False))


# Convert to a simple string Claude can read
data_context = high_risk[['sku_id','warehouse','avg_demand',
                           'last_inventory','weeks_of_supply',
                           'risk']].to_string(index=False)

print("High risk SKUs found:")
print(data_context)

# Now ask Claude about YOUR data
prompt = f"""Here is live inventory risk data from our SBD forecasting system:

{data_context}

Based on this data:
1. Which SKU and warehouse needs immediate attention?
2. What should the supply chain planner do right now?
3. Give a one line action for each HIGH risk item.

Be specific. Use the actual SKU names and numbers."""

message2 = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=500,
    system="You are a supply chain analyst at Stanley Black and Decker. Give specific actionable recommendations based only on the data provided.",
    messages=[{"role": "user", "content": prompt}]
)

print("\nClaude analyzing your SBD data:")
print(message2.content[0].text)
print(f"\nTokens used: {message2.usage.input_tokens} input, {message2.usage.output_tokens} output")


# ─────────────────────────────────────────
# PART 3 — CHROMADB VECTOR DATABASE
# ─────────────────────────────────────────
print("\n[PART 3] Building ChromaDB vector database...\n")

import chromadb

# Step 1 — Create ChromaDB client
# PersistentClient saves to disk so you don't rebuild every time
chroma_client = chromadb.PersistentClient(path="./vector_db")

# Delete old collection if exists — start fresh
try:
    chroma_client.delete_collection("sbd_forecasts")
except:
    pass

# Create new collection
collection = chroma_client.create_collection(name="sbd_forecasts")

print("ChromaDB collection created")

# Step 2 — Convert every SKU into a text document
# One document per SKU per warehouse = 80 documents total
documents = []
doc_ids   = []

for _, row in summary.iterrows():
    
    # Convert each row into a rich text description
    # The richer the text the better ChromaDB understands it
    doc = f"""
SKU: {row['sku_id']}
Warehouse: {row['warehouse']}
Brand: {'DEWALT' if row['sku_id'].startswith('D') else 'Stanley'}
Product type: {row['sku_id'].split('_')[1] if '_' in row['sku_id'] else 'Tool'}
Average weekly demand: {row['avg_demand']:.0f} units
Current inventory: {row['last_inventory']:.0f} units
Weeks of supply remaining: {row['weeks_of_supply']} weeks
Risk level: {row['risk']}
Status: {'Needs immediate reorder' if row['risk']=='HIGH' 
         else 'Monitor closely' if row['risk']=='MEDIUM' 
         else 'Stock levels adequate'}
""".strip()
    
    documents.append(doc)
    doc_ids.append(f"{row['sku_id']}_{row['warehouse']}")

# Step 3 — Store all documents in ChromaDB
collection.add(
    documents=documents,
    ids=doc_ids
)

print(f"Stored {len(documents)} documents in ChromaDB")
print(f"Saved to ./vector_db folder")



# ─────────────────────────────────────────
# PART 4 — FULL RAG PIPELINE
# ─────────────────────────────────────────
print("\n[PART 4] Full RAG pipeline — ask any question...\n")

def ask_sbd_data(question):
    """
    Full RAG pipeline.
    Takes any question in plain English.
    Finds relevant data from ChromaDB.
    Asks Claude to answer based on that data.
    Returns Claude's answer.
    """
    
    # Step 1 — Search ChromaDB for relevant documents
    results = collection.query(
        query_texts=[question],
        n_results=5
    )
    
    # Step 2 — Extract the retrieved documents
    retrieved_docs = results['documents'][0]
    
    # Step 3 — Join them into one context string
    context = "\n\n".join(retrieved_docs)
    
    # Step 4 — Build prompt with question and retrieved data
    prompt = f"""Here is relevant inventory data from our SBD forecasting system:

{context}

Question: {question}

Answer based only on the data provided above. 
Be specific — use actual SKU names, warehouses, and numbers."""

    # Step 5 — Ask Claude
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=400,
        system="You are a supply chain analyst at Stanley Black and Decker. Answer questions based only on the data provided.",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# Test with 3 different questions
questions = [
    "Which DEWALT tools are at highest risk?",
    "What is the inventory situation in Atlanta?",
    "Which Stanley products need attention?"
]

for q in questions:
    print(f"Question: {q}")
    print("Answer:")
    print(ask_sbd_data(q))
    print("-" * 50)