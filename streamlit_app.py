from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import streamlit as st
import os
import pickle
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from torch.nn.functional import embedding
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

llm= ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_retries=2
)


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_db_path = "drc_vector_db.pkl"



def create_db_vector(csv):
    loader = PyPDFLoader(file_path=csv)
    data = loader.load()
    #Storing the doc data into vectordb
    vector_db = FAISS.from_documents(data,embedding)
    # vector_db.save_local(vector_db_path)
    with open(vector_db_path,"wb") as file:
        pickle.dump(vector_db,file)

def get_qa_chain():
    #Loading local Vector db
    if os.path.exists(vector_db_path):
        with open(vector_db_path, "rb") as file:
            vector_db = pickle.load(file)

    # vector_db = FAISS.load_local(vector_db_path,embedding)

    #Creating a retriever
    retriever = vector_db.as_retriever()

    # The template
    template = """Given the following context and a question, generate an answer based on this context only.
       In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.(Don'ts : mention csv or its response field)
       If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

       CONTEXT: {context}

       QUESTION: {question}
       """

    prompt = PromptTemplate(
        input_variables=["content", "question"],
        template=template
    )

    #Creating a Retriever QA Chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain


# Designing UI

st.title("DRC Systems Policy Q&A")
btn = st.button("Create Knowledgebase")
query  = st.text_input("Enter Your Question :")


if __name__=="__main__":
    try:
        if btn:
            create_db_vector("DRC_details.pdf")

        chain = get_qa_chain()

        if query:
            #Getting response
            response = chain(query)
            answer = response["result"]
            st.header("**Answer:**")
            st.write(answer)
    except:
        st.write("Something went wrong! Please reload.")

    # create_db_vector("DRC_details.pdf")
    # chain = get_qa_chain()
    # print(chain("hello"))


