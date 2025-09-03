import os
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from dotenv import load_dotenv
from database import init_database, get_db_connection  # Assumes local module

# Initialize database 
if not init_database():
    raise RuntimeError("Failed to initialize database")

load_dotenv()

# Set your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize the model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

@tool
def get_inventory_data(product_name: str) -> str:
    """Get inventory data for a specific product from the database"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT product_name, current_stock, average_demand, lead_time
        FROM inventory
        WHERE product_name LIKE ?
    ''', (f'%{product_name}%',))

    result = cursor.fetchone()
    conn.close()

    if result:
        return f"Product: {result[0]}, Current Stock: {result[1]} units, Average Demand: {result[2]} units/day, Lead Time: {result[3]} days"
    else:
        return f"Product '{product_name}' not found in inventory database"

@tool
def get_all_inventory() -> str:
    """Get all inventory data from the database"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT product_name, current_stock, average_demand, lead_time
        FROM inventory
        ORDER BY product_name
    ''')

    results = cursor.fetchall()
    conn.close()

    if results:
        inventory_list = []
        for row in results:
            days_remaining = row[1] / row[2] if row[2] > 0 else 0
            status = "LOW STOCK" if days_remaining <= row[3] else "OK"
            inventory_list.append(f"{row[0]}: {row[1]} units, {days_remaining:.1f} days remaining ({status})")
        return "Current inventory status:\n" + "\n".join(inventory_list)
    else:
        return "No inventory data found"

# System prompt for RAG agent
system_prompt = """You are an inventory management expert with access to a real-time inventory database.

When a user asks about inventory decisions:
1. First, retrieve the current inventory data for the requested product using get_inventory_data
2. Analyze the data to determine if reordering is needed
3. Provide clear recommendations with reasoning

Key principles:
- If current stock will last ≤ lead time days, recommend reordering
- Calculate days remaining: current_stock ÷ average_demand
- Recommend ordering at least: average_demand × lead_time units
- For general inventory questions, use get_all_inventory to show overall status

Always base your analysis on the actual database data, not assumptions.
Be clear and concise in your recommendations."""

# Create agent
tools = [get_inventory_data, get_all_inventory]

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def inventory_chat(message, chat_history):
    """Handle chat messages with RAG"""
    if not message.strip():
        return chat_history, ""

    response = agent_executor.invoke({"input": message})
    return response["output"]

# Gradio interface
with gr.Blocks(title="RAG Inventory Management Assistant") as demo:
    gr.Markdown("# 📦 RAG Inventory Management Assistant")

    chatbot = gr.Chatbot(
        label="Inventory Assistant",
        height=400,
        type="messages"  
    )

    with gr.Row():
        msg = gr.Textbox(
            label="Ask about inventory",
            placeholder="e.g., 'Should I reorder toothpaste?'",
            lines=2,
            container=False,
        )
    submit_btn = gr.Button("Submit", variant="primary")
    clear = gr.Button("Clear Chat")

    with gr.Row():
        gr.Examples(
            examples=[
                "Should I reorder toothpaste?",
                "Check the inventory status for shampoo",
                "Show me all inventory with low stock",
            ],
            inputs=msg,
            label="Example Questions"
        )

    def respond(message, chat_history):
        if not message.strip():
            return chat_history, ""

        bot_message = inventory_chat(message, chat_history)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        return chat_history, ""

    msg.submit(respond, [msg, chatbot], [chatbot, msg])
    submit_btn.click(respond, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: [], None, [chatbot], queue=False)

if __name__ == "__main__":
    demo.launch()
