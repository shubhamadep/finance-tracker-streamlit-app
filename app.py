import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from streamlit_extras.add_vertical_space import add_vertical_space
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
from streamlit_chat import message

import streamlit as st
import boto3

access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
bucket_name = st.secrets["AWS_DEFAULT_REGION"]
OpenAI.api_key=st.secrets["OPENAI_API_KEY"]

model_name = "gpt-3.5-turbo"

st.set_page_config(page_title="Finance Chatbot", page_icon=":robot_face:")

class ChatApp:
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

    # generate a response
    def generate_response(self, query):
        st.session_state['messages'].append({"role": "user", "content": query})

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
            with get_openai_callback() as completion:
                response = chain.run(docs=docs, query=query)
                print(completion)

        st.session_state['messages'].append({"role": "assistant", "content": response})

        total_tokens = completion.total_tokens
        prompt_tokens = completion.prompt_tokens
        completion_tokens = completion.completion_tokens
        return response, total_tokens, prompt_tokens, completion_tokens
    
    ## Function for taking user provided prompt as input
    def get_text(self):
        input_text = st.text_input("You: ", "", key="input")
        return input_text

    def run(self):
        st.header("Ask your questions 💬")
        add_vertical_space(2)
        # container for text box
        container = st.container()

        # container for chat history
        response_container = st.container()

        # Initialise session state variables
        if 'generated' not in st.session_state:
            st.session_state['generated'] = []
        if 'past' not in st.session_state:
            st.session_state['past'] = []
        if 'messages' not in st.session_state:
            st.session_state['messages'] = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
        
        if 'model_name' not in st.session_state:
            st.session_state['model_name'] = []
        if 'cost' not in st.session_state:
            st.session_state['cost'] = []
        if 'total_tokens' not in st.session_state:
            st.session_state['total_tokens'] = []
        if 'total_cost' not in st.session_state:
            st.session_state['total_cost'] = 0.0

        # Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
        with st.sidebar:
            st.title("Currently using: " + model_name)
            counter_placeholder = st.empty()
            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
            clear_button = st.button("Clear Conversation", key="clear")
            add_vertical_space(1)
            st.write("Can answer questions about: ")
            st.write("Zomato Mar'23 earnings")
            st.write("Mahindra Mar'23 earnings")



        # reset everything
        if clear_button:
            st.session_state['generated'] = []
            st.session_state['past'] = []
            st.session_state['messages'] = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
            st.session_state['number_tokens'] = []
            st.session_state['model_name'] = []
            st.session_state['cost'] = []
            st.session_state['total_cost'] = 0.0
            st.session_state['total_tokens'] = []
            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_area("You:", key='input', height=100)
                submit_button = st.form_submit_button(label='Send')

        if user_input:
            with st.spinner("Working on it ..."):
                output, total_tokens, prompt_tokens, completion_tokens = self.generate_response(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.session_state['model_name'].append(model_name)
            st.session_state['total_tokens'].append(total_tokens)

            # from https://openai.com/pricing#language-models
            if model_name == "GPT-3.5":
                cost = total_tokens * 0.002 / 1000
            else:
                cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

            st.session_state['cost'].append(cost)
            st.session_state['total_cost'] += cost

            with response_container:
                if submit_button:
                    for i in range(len(st.session_state['generated'])-1, -1, -1):
                        message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                        message(st.session_state["generated"][i], key=str(i))
                        st.write(
                            f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
                        counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


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
            viewerBadge_container__1QSob {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)