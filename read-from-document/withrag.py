from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from datetime import datetime

persist_directory = 'vectorindex'
# # Invoke chain with RAG context
llm = Ollama(model="llama3")

if True:

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

vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OllamaEmbeddings(model="nomic-embed-text"))

## Prompt construction
prompt = ChatPromptTemplate.from_template(
    """
            Answer the following question only based on the given context
                                                    
            <context>
            {context}
            </context>
                                                    
            Question: {input}
"""
)

## Retrieve context from vector store
docs_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, docs_chain)

## Winner winner chicken dinner
response = retrieval_chain.invoke({"input": "If Vijay was born in 1975, what was his birth country Bharat or USA?"})
print(":::ROUND 2:::")
print(response["answer"])