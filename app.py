import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

import streamlit as st
import boto3

load_dotenv()
access_key = os.getenv("AWS_ACCESS_KEY_ID")
secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
bucket_name = os.getenv("BUCKET_NAME")

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

        # with st.sidebar:
        #     st.title('LLM Chat App')
        #     st.markdown('''
        #     ## About
        #     This app is an LLM-powered chatbot built using:
        #     - [Streamlit](https://streamlit.io/)
        #     - [LangChain](https://python.langchain.com/)
        #     - [OpenAI](https://platform.openai.com/docs/models) LLM model
        #     ''')
        #     add_vertical_space(5)

        #pdf = st.file_uploader("Upload your PDF", type='pdf')

        # if pdf is not None:
        #     self.process_pdf(pdf)

        query = st.text_input("")

        if query and self.vector_store is not None:
            docs = self.vector_store.similarity_search(query=query, k=3)

            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
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