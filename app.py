import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.prompts import (
    ChatPromptTemplate, 
    HumanMessagePromptTemplate,
)

from langchain.chains import LLMChain

import os

import streamlit as st
import boto3

access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
bucket_name = st.secrets["AWS_DEFAULT_REGION"]

OpenAI.api_key=st.secrets["OPENAI_API_KEY"]

class ChatApp:
    def check_password(self):
        """Returns `True` if the user had the correct password."""

        def password_entered():
            """Checks whether a password entered by the user is correct."""
            if st.session_state["password"] == st.secrets["password"]:
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # don't store password
            else:
                st.session_state["password_correct"] = False

            if "password_correct" not in st.session_state:
                # First run, show input for password.
                st.text_input(
                    "Password", type="password", on_change=password_entered, key="password"
                )
                return False
            elif not st.session_state["password_correct"]:
                # Password not correct, show input + error.
                st.text_input(
                    "Password", type="password", on_change=password_entered, key="password"
                )
                st.error("😕 Password incorrect")
                return False
            else:
                # Password correct.
                return True
    
    def __init__(self):
        self.vector_store = None
        # Create an S3 client
        # s3 = boto3.client('s3')
        s3 = boto3.client('s3')

        # Specify the bucket and file key (path) of the Pickle file
        bucket_name = 'pdfgptembeddings'
        file_key = 'pickle_file_05_25_23.pkl'

        # Download the file from S3
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        pickle_data = response['Body'].read()

        # Load the Pickle data
        self.vector_store = pickle.loads(pickle_data)

    def process_pdf(self, pdf):
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                self.vector_store = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            self.vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(self.vector_store, f)

    def run(self):
        st.header("Ask questions about your portfolio 💬")

        query = st.text_input("")

        if query and self.vector_store is not None:
            docs = self.vector_store.similarity_search(query=query, k=3)

            template = """
                You are a financial advisor that can answer questions based on the financial reports 
                and document provided: {docs}

                Only use factual information from the documents to answer the questions. 

                If you dont have enough information to answer the question, say "I don't have sufficient
                information to answer this question"
            """

            human_template = "Answer the following question: {query}"

            system_message_prompt = SystemMessagePromptTemplate.from_template(template)
            human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
            
            chat_prompt = ChatPromptTemplate.from_messages(
                [system_message_prompt, human_message_prompt]
            )

            llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0.2)
            chain = LLMChain(llm=llm, prompt=chat_prompt)
            with get_openai_callback() as cb:
                response = chain.run(docs=docs, query=query)
                print(cb)
            st.write(response)

def main():
    # streamlit_app.py
    def check_password():
        """Returns `True` if the user had the correct password."""

        def password_entered():
            """Checks whether a password entered by the user is correct."""
            if st.session_state["password"] == st.secrets["password"]:
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # don't store password
            else:
                st.session_state["password_correct"] = False

        if "password_correct" not in st.session_state:
            # First run, show input for password.
            st.text_input(
                "Password", type="password", on_change=password_entered, key="password"
            )
            return False
        elif not st.session_state["password_correct"]:
            # Password not correct, show input + error.
            st.text_input(
                "Password", type="password", on_change=password_entered, key="password"
            )
            st.error("😕 Password incorrect")
            return False
        else:
            # Password correct.
            return True

    if check_password():
        app = ChatApp()
        app.run()

if __name__ == '__main__':
    main()

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)