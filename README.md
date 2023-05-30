# finance-tracker-streamlit-app

MOST of the implementation can be found online, and is referenced from https://python.langchain.com/en/latest/index.html. Except the S3 part as the streamlit "data source" setup did not support .pkl file. 

https://streamlit.io/ is used for faster prototyping and is easy to use.

create-embeddings.py file creates indexes for embeddings from PDFs and stores it in a pickle file. ( https://github.com/facebookresearch/faiss )
We want these indexes created on top of the OpenAi Embeddings for the documents as we need to search for the closest documents in our document pools, and provide only those documents as context to
Chat GPT models ( 3.5, etc ) - More on this later. 

We then take this pickle file and add it in a AWS S3 bucket manually. TODO - use boto3 library to automate this if needed. 

app.py has the code which generates https://finance-chatbot.streamlit.app/. ( Reference API document: https://docs.streamlit.io/library/api-reference )
It uses boto3 to pull the pickle from our S3 bucket ( can use data loader from langchain which has the same implementation but poor documentation ) -> splits the document indexes into chunks ( currently set to 1000 ) -> depending on the prompt from the client query, search for the closest documents ( currently 4 closest documents are choosen )
, append them to a template prompt and send it to the GPT model -> pass the GPT's response to the client. 

As we are using Chat GPT 3.5 model which has a limit of 4000 tokens, we try to split the documents into chunks of 1000 tokens ( from which 200 are overlap to preserve context ) and pick 4 closest. This is tentetive and can be changed as needed. 

This app also has a simple auth to preserve mis-use of my personal OPENAI key. 
NOTE: personal keys might exists in the initial commits which have been deleted from their respective accounts. 
