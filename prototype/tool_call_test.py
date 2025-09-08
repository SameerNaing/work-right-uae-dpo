from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# Simple tool
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    print(f"🔧 TOOL EXECUTED: add_numbers({a}, {b})")
    return a + b

# Setup
llm = ChatOllama(model="mistral-nemo:12b-instruct-2407-q8_0", temperature=0.1,)
tools = [add_numbers]
agent = create_react_agent(llm, tools)

# Test input
user_message = "What is 5 + 3?"

print("=" * 50)
print("USER INPUT:", user_message)
print("=" * 50)

# Run agent and capture all messages
result = agent.invoke(
    {"messages": [HumanMessage(content=user_message)]},
    config={"configurable": {"thread_id": "test"}}
)

print("\n🔍 RAW MESSAGE FLOW:")
print("-" * 30)

for i, msg in enumerate(result["messages"], 1):
    print(f"\nStep {i}: {type(msg).__name__}")
    print(f"Content: {msg.content}")
    
    # THIS IS THE KEY PART - checking for tool_calls
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print(f"🚨 TOOL_CALLS DETECTED: {msg.tool_calls}")
        print("👆 This is what triggers LangGraph to execute tools!")
        
        # Show the exact structure that triggers tool execution
        for tool_call in msg.tool_calls:
            print(f"   - Tool Name: {tool_call['name']}")
            print(f"   - Arguments: {tool_call['args']}")
            print(f"   - Call ID: {tool_call['id']}")
    
    if hasattr(msg, 'tool_call_id'):
        print(f"Tool Result ID: {msg.tool_call_id}")

print("\n" + "=" * 50)
print("EXPLANATION:")
print("1. LLM generates AIMessage with tool_calls field")
print("2. LangGraph detects tool_calls in the message") 
print("3. LangGraph automatically routes to tools node")
print("4. Tools are executed and ToolMessage is created")
print("5. Process continues until no more tool_calls")
print("=" * 50)