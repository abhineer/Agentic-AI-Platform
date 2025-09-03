import os
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
    "You are an inventory management expert. Analyze inventory situations and provide clear reorder recommendations"
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

# Example usage
response = chain.invoke({
    "product": "toothpaste",
    "current_stock": 20,
    "average_demand": 5,
    "lead_time": 4
})

print(response.content)