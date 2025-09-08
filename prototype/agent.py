from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI



from tools import tools


llm = ChatOllama(model="qwen3:4b-instruct-2507-fp16", temperature=0.1,)

# llm = llm.bind_tools(tools=tools)
with open("./prompt-2.md") as f: 
    prompt = f.read()

agent = create_react_agent(
    model=llm,
    tools=tools, 
    prompt=prompt
)