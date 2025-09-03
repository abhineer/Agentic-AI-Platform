import os
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv

load_dotenv()   

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

# System prompt
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an inventory management expert. Analyze inventory situations and provide clear reorder recommendations with reasoning."
)

# User prompt template
user_prompt = HumanMessagePromptTemplate.from_template(
    """Should I reorder {product}? If so, how much?
    
Current stock: {current_stock} units
Average demand: {average_demand} units per day
Lead time: {lead_time} days

Provide your recommendation and reasoning."""
)

# Create chat prompt
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

# Create the chain
chain = chat_prompt | llm

def inventory_chat(product, current_stock, average_demand, lead_time):
    """Function to handle inventory analysis"""
    try:
        response = chain.invoke({
            "product": product,
            "current_stock": int(current_stock),
            "average_demand": float(average_demand),
            "lead_time": int(lead_time)
        })
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Inventory Management Assistant") as demo:
    gr.Markdown("# 📦 Inventory Management Assistant")
    gr.Markdown("Get AI-powered recommendations for inventory reordering decisions.")
    
    with gr.Row():
        with gr.Column():
            product_input = gr.Textbox(
                label="Product Name",
                placeholder="e.g., toothpaste",
                value="toothpaste"
            )
            stock_input = gr.Number(
                label="Current Stock (units)",
                value=20,
                minimum=0
            )
            demand_input = gr.Number(
                label="Average Daily Demand (units/day)",
                value=5.0,
                minimum=0.1
            )
            leadtime_input = gr.Number(
                label="Lead Time (days)",
                value=4,
                minimum=1
            )
            
            analyze_btn = gr.Button("Analyze Inventory", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="AI Recommendation",
                lines=8,
                interactive=False
            )
    
    # Connect the function to the button
    analyze_btn.click(
        fn=inventory_chat,
        inputs=[product_input, stock_input, demand_input, leadtime_input],
        outputs=output
    )
    
    # Add example
    gr.Examples(
        examples=[
            ["toothpaste", 20, 5.0, 4],
            ["shampoo", 15, 3.0, 5],
            ["soap", 50, 8.0, 3],
        ],
        inputs=[product_input, stock_input, demand_input, leadtime_input],
    )

if __name__ == "__main__":
    demo.launch()