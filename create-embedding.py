import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

from langchain.embeddings import OpenAIEmbeddings
import pickle
from langchain.vectorstores import FAISS

os.environ["OPENAI_API_KEY"] = "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"
root_dir = '/Users/shubhamadep/Desktop/workspace/pdf-chat-05-25-23'

# loader = TextLoader('single_text_file.txt')
loader = DirectoryLoader(f'{root_dir}/PDFs/', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
print(len(documents))

text_splitter = RecursiveCharacterTextSplitter(
                                               chunk_size=1000, 
                                               chunk_overlap=200)

texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

db_openAIEmbedd = FAISS.from_documents(texts, embeddings)
with open(f"pickle_file_05_25_23.pkl", "wb") as f:
    pickle.dump(db_openAIEmbedd, f)