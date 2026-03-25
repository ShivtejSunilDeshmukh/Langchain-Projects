import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
load_dotenv()
model=ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
    )
prompt = ChatPromptTemplate.from_messages([
    ('system', """
You are a movie data extractor. I will give you the name of a movie. Your task is to fetch details about the movie and return them **only in JSON format**. Do not include any text outside of JSON.  

The JSON should include the following fields:

{{
  "title": string,
  "release_year": number,
  "genre": [string],
  "director": string,
  "cast": [string],
  "plot_summary": string,
  "runtime_minutes": number,
  "language": string,
  "country": string,
  "rating": string
}}
    """),
('human',
 """
Extract information of movie based on this paragraph :
{paragraph}
""")
])
para=input("Give Para : ")
final_prompt = prompt.invoke(
    {"paragraph":para}
    )
response = model.invoke(final_prompt)
print(response.content)