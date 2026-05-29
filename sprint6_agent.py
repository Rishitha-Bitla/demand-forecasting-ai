"""
Sprint 6 — Supply Chain Agent
================================
An autonomous agent that checks all warehouses,
identifies risks, and generates a weekly reorder report.

Make sure your MCP server is running before running this:
uvicorn sprint5_mcp_server:app --reload --port 8000
"""

import os
import json
import requests
import anthropic
from dotenv import load_dotenv

load_dotenv()
claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MCP_URL = "http://localhost:8000"

print("=" * 55)
print("SPRINT 6 — Supply Chain Agent")
print("=" * 55)

# ─────────────────────────────────────────
# STEP 1 — Define tools Claude can use
# ─────────────────────────────────────────

tools = [
    {
        "name": "get_warehouse_summary",
        "description": "Get overall health summary for one warehouse. Use this first to understand the big picture for each warehouse.",
        "input_schema": {
            "type": "object",
            "properties": {
                "warehouse": {
                    "type": "string",
                    "description": "Warehouse name — Atlanta, Chicago, Dallas, or Newark"
                }
            },
            "required": ["warehouse"]
        }
    },
    {
        "name": "get_stockout_risk",
        "description": "Get top SKUs at highest stockout risk. Use this to identify which products need reordering urgently.",
        "input_schema": {
            "type": "object",
            "properties": {
                "warehouse": {
                    "type": "string",
                    "description": "Optional warehouse filter. Leave empty for all warehouses."
                },
                "top_n": {
                    "type": "integer",
                    "description": "How many SKUs to return. Default 5."
                }
            }
        }
    },
    {
        "name": "get_sku_forecast",
        "description": "Get detailed 4-week forecast for one specific SKU in one warehouse. Use this after identifying high risk SKUs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sku_id": {
                    "type": "string",
                    "description": "The SKU ID"
                },
                "warehouse": {
                    "type": "string",
                    "description": "Warehouse name"
                }
            },
            "required": ["sku_id", "warehouse"]
        }
    },
    {
        "name": "get_sales_history",
        "description": "Get recent sales history for one SKU. Use this to understand demand trends.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sku_id": {
                    "type": "string",
                    "description": "The SKU ID"
                },
                "warehouse": {
                    "type": "string",
                    "description": "Warehouse name"
                },
                "last_n_weeks": {
                    "type": "integer",
                    "description": "How many weeks of history. Default 12."
                }
            },
            "required": ["sku_id", "warehouse"]
        }
    }
]

print(f"Defined {len(tools)} tools for the agent")
print("Tools:", [t['name'] for t in tools])



# ─────────────────────────────────────────
# STEP 2 — Function that calls MCP server
# ─────────────────────────────────────────

def execute_tool(tool_name, tool_input):
    """
    When Claude decides to use a tool this function runs.
    It calls your MCP server and returns the result.
    
    tool_name  — which tool Claude wants to use
    tool_input — what parameters Claude wants to pass
    """
    
    print(f"  → Agent calling: {tool_name}({tool_input})")
    
    # Build the URL for this tool
    url = f"{MCP_URL}/tools/{tool_name}"
    
    # Call your MCP server
    response = requests.get(url, params=tool_input)
    
    if response.status_code == 200:
        result = response.json()
        print(f"  ← Got result: {str(result)[:80]}...")
        return json.dumps(result)
    else:
        error = f"Tool {tool_name} failed with status {response.status_code}"
        print(f"  ← Error: {error}")
        return json.dumps({"error": error})
    

# ─────────────────────────────────────────
# STEP 3 — The agent loop
# ─────────────────────────────────────────

def run_agent(goal):
    """
    Runs the agent with a given goal.
    
    The loop:
    1. Send goal + tools to Claude
    2. Claude decides — answer directly OR call a tool
    3. If tool — run it, send result back to Claude
    4. Claude decides again
    5. Repeat until Claude gives final answer
    """
    
    print(f"\nAgent goal: {goal}")
    print("=" * 55)
    
    # Conversation history — grows with every step
    messages = [{"role": "user", "content": goal}]
    
    step = 0
    max_steps = 15  # safety limit — stops infinite loops
    
    while step < max_steps:
        step += 1
        print(f"\nAgent step {step}:")
        
        # Send everything to Claude
        response = claude.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4000,
            system="""You are an autonomous supply chain agent at Stanley Black and Decker.
You have access to tools to check inventory risk across 4 warehouses.
Your goal is to complete the assigned task thoroughly.
Always check all 4 warehouses before writing your final report.
Use get_warehouse_summary first, then get_stockout_risk for urgent items,
then get_sku_forecast for HIGH risk SKUs.""",
            tools=tools,
            messages=messages
        )
        
        # Check what Claude decided
        if response.stop_reason == "end_turn":
            # Claude is done — extract final answer
            final = next(
                (b.text for b in response.content if hasattr(b, 'text')), 
                "No response gener" \
                "ated"
            )
            print("\nAgent complete — writing final report")
            return final
        
        elif response.stop_reason == "tool_use":
            # Claude wants to use a tool
            # Add Claude's response to history
            messages.append({
                "role": "assistant",
                "content": response.content
            })
            
            # Find all tool calls in this response
            tool_results = []
            
            for block in response.content:
                if block.type == "tool_use":
                    
                    # Execute the tool
                    result = execute_tool(block.name, block.input)
                    
                    # Collect result
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            
            # Send all tool results back to Claude
            messages.append({
                "role": "user",
                "content": tool_results
            })
        
        else:
            print(f"Unexpected stop reason: {response.stop_reason}")
            break
    
    return "Agent reached maximum steps"


# ─────────────────────────────────────────
# STEP 4 — Give the agent a goal and run it
# ─────────────────────────────────────────

goal = """You are the weekly supply chain agent for Stanley Black and Decker.

Your task:
1. Check all 4 warehouses — Atlanta, Chicago, Dallas, Newark
2. Identify which warehouses need attention
3. Find the top HIGH risk SKUs across all warehouses
4. For each HIGH risk SKU get the detailed forecast
5. Write a complete Weekly Reorder Report

The report should include:
- Executive summary of warehouse health
- List of SKUs that need immediate reorder with quantities
- Recommended actions for each warehouse
- Priority ranking of what to action first"""

# Run the agent
final_report = run_agent(goal)

print("\n" + "=" * 55)
print("FINAL WEEKLY REORDER REPORT")
print("=" * 55)
print(final_report)