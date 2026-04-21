# AutoStream Conversational AI Agent

This project is a Conversational AI Agent built for **AutoStream**, a fictional SaaS product providing automated video editing tools for content creators. The agent processes casual greetings, handles product & pricing inquiries using a local RAG Knowledge Base, and qualifies high-intent leads to capture their details via a mock tool execution.

## 1. How to run the project locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Abhiprameesh/Social-to-Lead-Agentic-Workflow.git
   cd Social-to-Lead-Agentic-Workflow
   ```
2. **Install the required dependencies:**
   Make sure you have Python 3.9+ installed.
   ```bash
   pip install -r requirements.txt
   ```
3. **Set your Gemini API Key:**
   Create a `.env` file in the root directory and add your Google Gemini API key:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```
4. **Run the Interactive Agent:**
   ```bash
   python agent.py
   ```
   *You can interact with the agent in the terminal. Try querying about the Pro plan or indicating you want to sign up for your YouTube channel.*

## 2. Architecture Explanation

For this agent, **LangGraph** was selected over AutoGen or traditional LangChain Chains/Agents because of its explicit control over state and inherently cyclical structure. LangGraph operates as a state machine where execution can route dynamically based on conditions, making it perfectly suited for robust intent-routing and human-in-the-loop interventions (like asking missing lead information before tool execution). 

The state is managed using a custom Pydantic `AgentState` typed dictionary. It stores:
1. `messages`: A list of conversation history utilizing LangGraph's `add_messages` reducer to append subsequent messages natively.
2. `intent`: The output of a structure-enforced LLM classification (greeting, inquiry, high-intent).
3. `lead_*` variables: Name, Email, and Platform.

On every user message, the graph's first node classifies the intent and extracts potential lead variables from context, seamlessly merging them into the global state. Based on the intent, the execution routes to dedicated handler nodes: returning friendly chat for greetings, using a local FAISS retriever for RAG-based context answering, or verifying lead details. If high-intent variables are complete, the node fires the `mock_lead_capture` tool; otherwise, it asks the user for the missing fields, demonstrating clean state retention across multiple conversational turns.

## 3. WhatsApp Deployment Question

**How would you integrate this agent with WhatsApp using Webhooks?**

Integrating this LangGraph agent with WhatsApp is straightforward using a platform like Twilio or Meta's official WhatsApp Business API. 

1. **Webhook Endpoint Setup:** You would wrap the `app.invoke()` logic in a web framework like FastAPI or Flask. The application would expose a POST `/webhook` route. 
2. **Meta/Twilio Configuration:** In the WhatsApp business dashboard, you configure the webhook URL to point to your deployed server. Now, every time a user sends a WhatsApp message, Meta sends a JSON payload to your server containing the `From` phone number and the text body.
3. **Session Management (State):** Because LangGraph requires maintaining the Conversation State buffer across messages (unlike terminal variables), you would map the WhatsApp sender's phone number to an ongoing session ID. A database (e.g., PostgreSQL or Redis) or LangGraph's built-in `MemorySaver()` checkpointer can store and retrieve the `AgentState` corresponding to that specific phone number.
4. **Agent Execution:** The incoming text traverses your LangGraph logic, accumulating messages and intent status.
5. **Dispatching Replies:** Once the agent formulates a response, the backend makes an HTTP POST request back to the WhatsApp Business API endpoint with the generated reply. The sender receives the AI response natively on their phone.
