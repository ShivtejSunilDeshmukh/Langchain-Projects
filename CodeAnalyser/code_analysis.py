import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_react_agent,AgentExecutor
from langchain.tools import tool

load_dotenv()

llm =ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    model="openrouter/free"
)

@tool
def run_code(code:str):
    """
    use this to execute the code and check if the output is correct and as expected
    """
    try:
        local_vars={}
        exec(code,{},local_vars)
        return str(local_vars)
    except Exception as e:
        return str(e)

@tool
def analyze_code(code:str):
    """Analyze Python code for:
    - performance issues
    - readability
    - potential bugs
    - suggest improved versions"""
    prompt=f"""analyze this {code} and suggest improvement"""
    return llm.invoke(prompt).content
    

tools=[run_code,analyze_code]

custom_prompt_template=PromptTemplate(
    input_variables=["input","tools","tool_names","agent_scratchpad"],
    template="""
    You are an AI assistant that analyzes and debugs Python code.

You have access to the following tools:

{tools}

Tool names:
{tool_names}

When solving the problem, follow this format exactly:

Thought: think about what to do
Action: one of [{tool_names}]
Action Input: the code
Observation: result of the tool

You may repeat this multiple times.

When you are done:

Thought: I now know the final answer
Final Answer: explanation to the user

User code:
{input}

{agent_scratchpad}
    """
)
agent_1=create_react_agent(
    llm=llm,
    tools=tools,
    prompt=custom_prompt_template
)
agent_0=AgentExecutor(
    agent=agent_1,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    # max_iterations=3
)

while True:
    inp=input("Provide Pyhton Code : ")
    if inp=="0":
        break
    response=agent_0.invoke({"input": inp})
    print(response)