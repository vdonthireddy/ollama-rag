from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Simple chain invocation
## LLM + Prompt
llm = Ollama(model="llama3")
output = StrOutputParser()
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a skilled assistant.",
        ),
        ("human", "{user_input}"),
    ]
)
chain = prompt | llm | output

## Winner winner chicken dinner
response = chain.invoke({"user_input": "If Vijay was born in 1975, what was his birth country?"})
print(":::ROUND 1:::")
print(response)