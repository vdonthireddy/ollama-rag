from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from datetime import datetime

# import bs4
# from langchain_community.document_loaders import WebBaseLoader

# # LOAD the data from a web url
# bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs={"parse_only": bs4_strainer},
# )
print('started at: ' + str(datetime.now()))

persist_directory = 'vectorindex'

if True: # After the first time (once the document is loaded, split and vectors saved in vectordb), you can set this condition to False 

    # from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.document_loaders import TextLoader

    # loader = PyPDFLoader("data/India.txt")
    loader = TextLoader("data/Bharat.txt")
    # pages = loader.load_and_split()

    docs = loader.load()

    print('loaded.: ' + str(datetime.now()))

    # SPLIT data into chunks before storing them in vector db
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    print('split complete.: ' + str(datetime.now()))

    # STORE the embeddings (array of ints) in vector db
    vectorstore = Chroma.from_documents(
        documents=all_splits, 
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory = persist_directory,
    )

    print('stored.: ' + str(datetime.now()))

#RETRIEVER: 
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# template = """Use the following pieces of context to answer the question at the end.
# Please provide answer only from my document(s) provided.
# Always say "thanks for asking!" at the end of the answer.

template = """Use the following pieces of context to answer the question at the end.
Please answer from my document(s) only.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""

from langchain_core.prompts import PromptTemplate
custom_rag_prompt = PromptTemplate.from_template(template)

from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | Ollama(model="llama3")
    | StrOutputParser()
)

# response = rag_chain.invoke("What is the capital of USA?")
# response = rag_chain.invoke("When did Bharat get independence")
# response = rag_chain.invoke("Vijay is the citizen of which country in 2024?")
# response = rag_chain.invoke("Vijay is the citizen of which country in 1978?")
# response = rag_chain.invoke("What is the capital of Bharat?")
# response = rag_chain.invoke("which country is to the north of Bharat?")
# response = rag_chain.invoke("which country is to the north of Bharat?")
# response = rag_chain.invoke("What is the most popular sport?")
# response = rag_chain.invoke("What is Bharat famous for?")
# response = rag_chain.invoke("Where was Vijay born")
response = rag_chain.invoke("If Vijay was born in 1975, what was his birth country Bharat or USA?")

print(response)
print('ended at: ' + str(datetime.now()))
