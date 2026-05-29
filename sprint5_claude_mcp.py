import os
import requests
import anthropic
from dotenv import load_dotenv

load_dotenv()
claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MCP_URL = "http://localhost:8000"

print("=" * 55)
print("SPRINT 5 — Claude connected to MCP Server")
print("=" * 55)

def call_mcp_tool(tool_name, params):
    """
    Calls your MCP server tool and returns the result.
    This is how Claude gets data from your server.
    """
    url = f"{MCP_URL}/tools/{tool_name}"
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Tool failed: {response.status_code}"}

def ask_claude_with_mcp(question):
    """
    1. Decides which MCP tool to call based on the question
    2. Calls the tool and gets real data
    3. Passes data to Claude
    4. Returns Claude's answer
    """
    print(f"\nQuestion: {question}")
    print("-" * 40)
    
    # Step 1 — get warehouse summary for all warehouses
    atlanta = call_mcp_tool("get_warehouse_summary", {"warehouse": "Atlanta"})
    chicago = call_mcp_tool("get_warehouse_summary", {"warehouse": "Chicago"})
    dallas  = call_mcp_tool("get_warehouse_summary", {"warehouse": "Dallas"})
    newark  = call_mcp_tool("get_warehouse_summary", {"warehouse": "Newark"})
    
    # Step 2 — get stockout risk across all warehouses
    risks = call_mcp_tool("get_stockout_risk", {"top_n": 5})
    
    # Step 3 — build context from MCP data
    context = f"""
WAREHOUSE SUMMARIES:
Atlanta:  {atlanta['summary']}
Chicago:  {chicago['summary']}
Dallas:   {dallas['summary']}
Newark:   {newark['summary']}

TOP 5 MOST URGENT SKUs ACROSS ALL WAREHOUSES:
{risks['summary']}
Details: {risks['results']}
"""
    
    # Step 4 — ask Claude
    response = claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        system="You are a supply chain analyst at Stanley Black and Decker. Answer based only on the data provided.",
        messages=[{
            "role": "user",
            "content": f"Here is live data from our MCP server:\n{context}\n\nQuestion: {question}"
        }]
    )
    
    print("Claude's answer:")
    print(response.content[0].text)
    print(f"\nTokens: {response.usage.input_tokens} in, {response.usage.output_tokens} out")

# Test with real questions
ask_claude_with_mcp("Which warehouse needs the most attention right now?")
ask_claude_with_mcp("What are the top 3 SKUs I should reorder today?")