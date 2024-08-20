import streamlit as st
from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

SEN_TR_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "llama3:8b"



def get_response(user_input):
    retriever_chain = get_context_retrieval_chain(st.session_state.vector_store)
    conversational_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversational_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_query
        })
    return response['answer']

def get_context_retrieval_chain(vector_store):
    llm = Ollama(model=LLM_MODEL)
    retriever = vector_store.as_retriever()
    prompt  = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}"),
        ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation ")
    ])
    
    retriever_chain = create_history_aware_retriever(llm,retriever,prompt)
    return retriever_chain

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    vector_store = FAISS.from_documents(document_chunks,HuggingFaceEmbeddings(model_name=SEN_TR_MODEL))
    return vector_store
    
def get_conversational_rag_chain(retriever_chain):
   llm = Ollama(model=LLM_MODEL) 
   prompt  = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("system","Answer the users question based on the below context:\n\n{context}"),
        ("user","{input}")
    ]) 
   stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
   return create_retrieval_chain(retriever_chain, stuff_documents_chain)
    
#app configuration
st.set_page_config(page_title="conversational chatbot")
st.title("Conversational Chatbot") 

#sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")
if website_url is None or website_url =="":
    st.info ("enter url")

else:
    #session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
        AIMessage(content = "Welcome")
    ]
    if "vector_store" not in st.session_state:   
        st.session_state.vector_store = get_vectorstore_from_url(website_url)
    
    #user_input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query!="":
        response = get_response(user_query)
        
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
        
    #conversation front end
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
    