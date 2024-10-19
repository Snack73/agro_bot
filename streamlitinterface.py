import os
import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain




os.environ["GOOGLE_API_KEY"] = "AIzaSyAMmDwNe5lGXE8VALHRye1XdL885dArAYM"

st.header("FarmoBoy")
with st.sidebar:
    st.title("FarmoBot using Gemini-Pro")
    add_vertical_space(3)
    st.write("Made by Aayush Roy")
    add_vertical_space(3)
    st.write('''
Made using:\n
-Langchain\n
-Streamlit\n
-Google Gemini\n
                 ''')

def main():
    pdf_docs = st.file_uploader("Submit you PDF File here", type = "pdf")

    if pdf_docs is not None:
        pdf_reader = PdfReader(pdf_docs)
        text = " "
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator = '\n',
            chunk_size = 800,
            chunk_overlap = 200,
            length_function = len
        )

        texts = text_splitter.split_text(text=text)

        embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        vectorstore = FAISS.from_texts(texts, embedding=embeddings)

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        prompt = PromptTemplate.from_template('''
        Answer the question based only on the provided context. If you do not know the answer say "I can't seem to find an answer for your question from the provided data".
        Always say "Pleased to provide the answer for your question" whenever you know the answer in the starting.                                                                         
        <context>                                    
        {context}
        </context>
        
        Question : {input}                               
                                              
        ''')

        
        llm = ChatGoogleGenerativeAI(model = "gemini-pro", temperature = 0.3)


        document_chain = create_stuff_documents_chain(llm, prompt)

        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        user_input = st.text_input("Ask questions about your PDF file:")

        if user_input:
            ans = retrieval_chain.invoke({"input": user_input})
            st.write(ans['answer'])
            





if __name__ == "__main__":
    main()