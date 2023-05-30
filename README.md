# finance-tracker-streamlit-app

MOST of these implementation can be found online, and is taken from https://python.langchain.com/en/latest/index.html. Except the S3 part as the streamlit data source setup did not support .pkl file. 

https://streamlit.io/ is used for faster prototyping data based apps.

create-embeddings.py file creates embeddings from PDFs and stores it in a pickle file. https://github.com/facebookresearch/faiss
We want these indexes created on top of the OpenAi Embeddings for the documents as we need to search for the closest documents in our document pools, and provide only those documents as context to
Chat GPT models ( 3.5, etc ). 

We then take this pickle file and add it in a AWS S3 bucket manually. TODO - use boto3 library to automate this. 

app.py has the code which generates https://finance-chatbot.streamlit.app/. ( Reference API document: https://docs.streamlit.io/library/api-reference )
It uses boto3 to pull the pickle from our S3 bucket ( also can use data loader from langchain which has the same implementation but poor documentation ) -> splits the indexes created into chunks ( currently set to 1000 ) -> depending on the prompt from the client query search the closest documents
and append them to a template prompt and send it to the GPT model -> revert back the response to the client. 

As we are using Chat GPT 3.5 model which has a limit of 4000 tokens, we try to split the documents into 1000 ( from which 200 are overlap to preserve context ). This is tentetive and can be changed
as needed. 

This app also has a simple auth to preserve mis use of my personal OPENAI key. 


NOTE: personal keys might exists in the initial commits which have been deleted from accounts. 
