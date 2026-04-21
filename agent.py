import os
from typing import Annotated, TypedDict, Literal, Optional
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Required Mock Tool Function
def mock_lead_capture(name: str, email: str, platform: str):
    """Mock tool to capture lead data."""
    print("\n" + "="*50)
    print(f"✅ TOOL EXECUTION: mock_lead_capture()")
    print(f"Lead captured successfully: {name}, {email}, {platform}")
    print("="*50 + "\n")


# 1. Define the State
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: Optional[str]
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool


# 2. Pydantic Model for Intent and Extraction
class IntentAndExtraction(BaseModel):
    intent_type: Literal["greeting", "product_inquiry", "high_intent"] = Field(
        description="Classify the user intent based on the conversation history."
    )
    extracted_name: Optional[str] = Field(default=None, description="The user's name, if provided.")
    extracted_email: Optional[str] = Field(default=None, description="The user's email address, if provided.")
    extracted_platform: Optional[str] = Field(default=None, description="The user's creator platform (e.g., YouTube), if provided.")


# Global variables for RAG and LLM
retriever = None
llm = None

def setup():
    """Initializes LLM and the RAG pipeline."""
    global retriever, llm
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

    # Initialize RAG Pipeline
    kb_path = "knowledge_base.md"
    if os.path.exists(kb_path):
        loader = TextLoader(kb_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    else:
        print("Warning: knowledge_base.md not found!")

# Graph Nodes

def classify_and_extract(state: AgentState):
    """Analyzes the latest message to classify intent and extract lead details."""
    messages = state.get("messages", [])
    
    # Use structured output to get intent and extractions
    analyzer_llm = llm.with_structured_output(IntentAndExtraction)
    
    system_prompt = (
        "You are an analyzer for AutoStream, an AI video editing SaaS.\n"
        "Analyze the user's latest latest message and the conversation history.\n"
        "1. Classify the intent into one of: 'greeting', 'product_inquiry', 'high_intent'.\n"
        "   - 'high_intent' means the user wants to sign up, buy a plan, or try the product.\n"
        "2. Extract their name, email, and creator platform if they mentioned them."
    )
    
    response = analyzer_llm.invoke([SystemMessage(content=system_prompt)] + messages)
    
    # Merge extracted details into state without overwriting existing ones with None
    updates = {"intent": response.intent_type}
    if response.extracted_name: updates["lead_name"] = response.extracted_name
    if response.extracted_email: updates["lead_email"] = response.extracted_email
    if response.extracted_platform: updates["lead_platform"] = response.extracted_platform
    
    # Ensure lead_captured is in state
    if "lead_captured" not in state:
        updates["lead_captured"] = False
        
    print(f"\n[DEBUG - Detector] Intent classified as: {response.intent_type}")
    
    return updates


def route_intent(state: AgentState) -> str:
    """Routes to the appropriate handler node based on intent."""
    intent = state.get("intent")
    if intent == "high_intent":
        return "handle_lead"
    elif intent == "product_inquiry":
        return "handle_inquiry"
    else:
        return "handle_greeting"


def handle_greeting(state: AgentState):
    """Handles casual greetings."""
    messages = state.get("messages", [])
    prompt = "You are a friendly agent for AutoStream. The user just greeted you. Respond warmly and briefly, then ask how you can help."
    response = llm.invoke([SystemMessage(content=prompt)] + messages)
    return {"messages": [response]}


def handle_inquiry(state: AgentState):
    """Handles RAG pipeline for product / pricing inquiries."""
    messages = state.get("messages", [])
    last_message = messages[-1].content if messages else ""
    
    # Retrieve knowledge base chunks
    docs = retriever.invoke(last_message) if retriever else []
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = (
        "You are a helpful customer support agent for AutoStream, an AI video editing SaaS. "
        "Answer the user's question accurately using ONLY the provided Knowledge Base information below. "
        "Do not hallucinate features or pricing that are not listed.\n\n"
        f"KNOWLEDGE BASE CONTEXT:\n{context}\n"
    )
    
    response = llm.invoke([SystemMessage(content=prompt)] + messages)
    return {"messages": [response]}


def handle_lead(state: AgentState):
    """Handles high intent users, collects missing details, and executes lead capture."""
    messages = state.get("messages", [])
    name = state.get("lead_name")
    email = state.get("lead_email")
    platform = state.get("lead_platform")
    lead_captured = state.get("lead_captured", False)
    
    if lead_captured:
        return {"messages": [AIMessage(content="We already captured your lead! We will touch base with you soon. Anything else you need?")]}
    
    missing_fields = []
    if not name: missing_fields.append("Name")
    if not email: missing_fields.append("Email")
    if not platform: missing_fields.append("Creator Platform (like YouTube, TikTok, etc.)")
    
    if missing_fields:
        prompt = (
            "You are a customer success agent for AutoStream. The user wants to sign up. "
            f"You MUST ask them to provide their missing details politely: {', '.join(missing_fields)}. "
            "Do not ask for anything else."
        )
        response = llm.invoke([SystemMessage(content=prompt)] + messages)
        return {"messages": [response]}
    else:
        # ALL DETAILS COLLECTED -> FIRE TOOL EXECUTION
        mock_lead_capture(name, email, platform)
        success_msg = f"Awesome {name}! I have successfully registered your interest for your {platform} channel using {email}. Our team will contact you shortly!"
        return {
            "messages": [AIMessage(content=success_msg)],
            "lead_captured": True
        }


# 3. Build the Graph
def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("classify", classify_and_extract)
    workflow.add_node("handle_greeting", handle_greeting)
    workflow.add_node("handle_inquiry", handle_inquiry)
    workflow.add_node("handle_lead", handle_lead)
    
    # Add edges
    workflow.add_edge(START, "classify")
    
    workflow.add_conditional_edges(
        "classify",
        route_intent,
        {
            "handle_greeting": "handle_greeting",
            "handle_inquiry": "handle_inquiry",
            "handle_lead": "handle_lead"
        }
    )
    
    workflow.add_edge("handle_greeting", END)
    workflow.add_edge("handle_inquiry", END)
    workflow.add_edge("handle_lead", END)
    
    return workflow.compile()


def run_interactive():
    from dotenv import load_dotenv
    load_dotenv()
    
    if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
        print("Please set GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable. You can create a .env file.")
        return
        
    # Some older langchain modules explicitly look for GOOGLE_API_KEY
    if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
        
    setup()
    app = build_graph()
    
    print("\n--- AutoStream Conversational Agent ---")
    print("Type 'exit' or 'quit' to stop.\n")
    
    # Initialize state
    state = {
        "messages": [],
        "lead_captured": False
    }
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break
                
            state["messages"].append(HumanMessage(content=user_input))
            
            # The app.invoke handles the state updates
            new_state = app.invoke(state)
            
            # The newest message from the AI is the last one
            raw_content = new_state["messages"][-1].content
            
            # Gemini sometimes returns content as a list of blocks instead of a plain string
            if isinstance(raw_content, list):
                ai_text = "".join([block.get("text", "") for block in raw_content if isinstance(block, dict)])
            else:
                ai_text = raw_content
                
            print(f"\nAgent: {ai_text}\n")
            
            # We must pass the updated state to the next iteration to retain memory
            state = new_state
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    run_interactive()
