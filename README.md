****Inventory Management Agent****
This project is a , AI-powered inventory management assistant built with LangChain, OpenAI's GPT models, and a Gradio user interface. It leverages a Retrieval-Augmented Generation (RAG) architecture to answer natural language queries about inventory by querying a real-time database.

The assistant is enhanced with robust security guardrails to prevent prompt injection and other misuse, and an intelligent caching layer to ensure high performance and reduce redundant API calls.

✨** Key Features**
**Core Functionality**
Conversational AI Agent: Uses a LangChain agent to understand user queries and interact with backend tools.

**Real-time Database Integration:** Connects to a local inventory database to provide up-to-date stock information.

**Intelligent Tooling: **The agent can:

get_inventory_data: Fetch stock levels, demand, and lead time for a specific product.

get_all_inventory: Provide a summary of all items in the inventory.

calculate_reorder_quantity: Analyze stock data to recommend if and when to reorder a product.

**Email Integration:** Includes a utility to send email alerts directly from the UI.

**🔒 Security & Performance**
Custom Input Guardrails: A built-in Guardrails class provides a critical layer of security by:

Blocking Prompt Injections: Detects and rejects malicious inputs designed to manipulate the LLM.

Preventing Role Switching: Stops attempts to make the agent change its designated role.

Protecting System Instructions: Denies requests trying to reveal the underlying system prompt.

Validating Input: Checks for suspicious patterns and excessive length.

**Intelligent Caching:** A custom InputCache class dramatically improves performance:

Faster Responses: Caches responses to frequently asked questions, providing near-instantaneous answers.

Reduced Costs: Minimizes calls to the OpenAI API.

Time-To-Live (TTL): Cache entries automatically expire (default: 5 minutes) to ensure data freshness.

Real-time Stats: The UI displays live cache statistics, including hit rate, misses, and current size.

**UI & Observability**
Interactive Web UI: A clean and user-friendly interface built with Gradio.

Example Prompts: Helps users get started quickly with pre-built example questions.

Opik Tracing: Integrated with OpikTracer for observability and debugging of agent execution flows.

🏛️** Architecture**
The application follows a modern RAG agent architecture:

User Input: A user asks a question through the Gradio UI (e.g., "Should I reorder toothpaste?").

Guardrail Evaluation: The input is first sanitized and validated by the Guardrails class. If the input is unsafe, it is rejected immediately.

Cache Check: If the input is valid, the InputCache checks if an identical query has been answered recently. If a fresh, cached response exists, it's returned instantly (marked with a 🔄 emoji).

Agent Execution: If the query is not in the cache (a "cache miss"), it's passed to the LangChain AgentExecutor.

Tool Selection & Execution: The agent, guided by its system prompt, determines the best tool to use. It might call calculate_reorder_quantity('toothpaste').

Database Query: The selected tool connects to the SQLite database, retrieves the necessary data (current stock, demand, lead time), and performs its calculation.

Response Generation: The data returned from the tool is passed back to the LLM, which formulates a human-readable, natural language response.

Update Cache & UI: The final response is stored in the cache for future use and is displayed to the user in the chatbot interface.
