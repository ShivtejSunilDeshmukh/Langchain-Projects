import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools import PythonREPLTool
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.chains import LLMChain
from langchain_classic.memory import ConversationBufferMemory
import streamlit as st
import datetime

# ===== Helper function to save report =====
def save_report(topic, report_text, folder="Copilot/doc"):
    os.makedirs(folder, exist_ok=True)
    safe_topic = topic.replace(" ", "_").replace("/", "_")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_topic}_{timestamp}.txt"
    filepath = os.path.join(folder, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)
    return filepath

# ===== Load environment variables =====
load_dotenv()

# ===== LLM Setup =====
llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# ===== Prompts =====
prompt_0 = PromptTemplate(
    input_variables=["topic", "context"],
    template="""
You are an expert AI research assistant.

Your job is to analyze the given research topic using the provided information and produce a comprehensive research report.

Research Topic:
{topic}

Collected Information:
{context}

Instructions:
- Carefully analyze the collected information.
- Combine insights and remove redundant information.
- Write a clear, well-structured research report.
- Use professional language.
- If information is missing, infer reasonable insights based on general knowledge.
- Cite sources when possible.

Generate the report in the following structure:

Title: {topic}

1. Overview
Provide a clear introduction explaining the topic and why it is important.

2. Key Concepts
Explain the fundamental ideas related to the topic.

3. Market Trends / Recent Developments
Describe current developments, trends, or technological progress.

4. Technical or Industry Challenges
Explain major limitations, risks, or barriers.

5. Future Outlook
Discuss future potential, growth opportunities, and predictions.

6. Conclusion
Provide a concise summary of the research.

Make the report detailed, insightful, and easy to read.
"""
)

custom_react_prompt = PromptTemplate(
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
    template="""
You are an AI research assistant. Your job is to answer the user's question or perform the requested task.
You have access to the following tools:

{tools}

Tool Names: {tool_names}

Instructions:
- Think step by step before using a tool.
- Use the appropriate tool when needed.
- Show your reasoning in the scratchpad before taking action.
- When done, give the final answer clearly.

User Input:
{input}

Previous Reasoning:
{agent_scratchpad}

Begin reasoning and tool usage:
"""
)

# ===== Tools =====
search = DuckDuckGoSearchRun()
web_tool = Tool(
    name="Web Search",
    func=search.run,
    description="Search the internet for current information about companies, technologies, news, and research topics."
)
python_tool = Tool(
    name="Python Calculator",
    func=PythonREPLTool().run,
    description="Useful for performing mathematical calculations, statistics, and data analysis using Python."
)
tools = [web_tool, python_tool]

# ===== Agent Setup =====
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
react_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=custom_react_prompt
)
agent_executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    memory=memory,
    verbose=False,
    handle_parsing_errors=True
)

# ===== Report Chain =====
report_chain = LLMChain(
    llm=llm,
    prompt=prompt_0
)

# ===== Streamlit UI =====
st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("AI Research Assistant")
st.write("Enter a research topic or question, and the AI will generate a detailed report.")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    folder = st.text_input("Save reports folder", value="Copilot/doc")
    verbose_mode = st.checkbox("Show agent reasoning", value=True)

# Input for query
query = st.text_input("Enter your research topic or question:")

if st.button("Generate Report"):
    if not query.strip():
        st.warning("Please enter a topic or question first.")
    else:
        with st.spinner("Running AI agent..."):
            # Run agent
            response = agent_executor.invoke({"input": query})
            # Safely handle different LangChain return types
            agent_output = response.get("output") or response.get("output_text") or str(response)
            
            # Generate report
            report = report_chain.run({
                "topic": query,
                "context": agent_output
            })
            
            # Save report
            path = save_report(query, report, folder=folder)

        st.success(f"Report saved to: {path}")

        # Tabs for agent reasoning and final report
        tabs = st.tabs(["Agent Reasoning", "Final Report"])
        with tabs[0]:
            if verbose_mode:
                st.subheader("🛠 Agent Raw Output")
                st.code(agent_output)
            else:
                st.info("Verbose mode is off. Enable in the sidebar to view agent reasoning.")

        with tabs[1]:
            st.subheader("Generated Research Report")
            st.text_area("Report", report, height=400)
            st.download_button(
                label="Download Report as TXT",
                data=report,
                file_name=f"{query.replace(' ','_')}.txt",
                mime="text/plain"
            )