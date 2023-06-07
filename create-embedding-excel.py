import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Set the OPENAI_API_KEY from local environment variables
api_key = os.environ.get("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

# loader = TextLoader('single_text_file.txt')
loader = DirectoryLoader(f'./XLs/', glob="./*.xlsx", loader_cls=UnstructuredExcelLoader)
documents = loader.load()
print(len(documents))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

db_openAIEmbedd = FAISS.from_documents(texts, embeddings)
with open(f"pickle_file_xlsx_06_06_23.pkl", "wb") as f:
    pickle.dump(db_openAIEmbedd, f)
