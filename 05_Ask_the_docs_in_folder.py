import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from pypdf import PdfReader
import docx
from apikey import apikey


# Set your OpenAI API key here
openai_api_key= apikey

# Function to read .txt, .pdf, and .docx files
def read_file(file_path):
    file_extension = os.path.splitext(file_path)[1]
    text = ""

    if file_extension == ".txt":
        with open(file_path, "r") as file:
            text = file.read()
    elif file_extension == ".pdf":
        with open(file_path, "rb") as file:
            pdf_reader = PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num] .extract_text()
    elif file_extension == ".docx":
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text

    return text

def generate_response(file_paths, openai_api_key, query_text):
    # Load documents
    documents = [read_file(file_path) for file_path in file_paths]

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(documents)

    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Create a vectorstore from documents
    db = Chroma.from_documents(texts, embeddings)

    # Create retriever interface
    retriever = db.as_retriever()

    # Create QA chain
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)

    return qa.run(query_text)

# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App', layout='wide', initial_sidebar_state='expanded')
st.title('🦜🔗 Ask the Doc App')

# Folder path input
folder_path = st.text_input('Enter the folder path:', '')

# Get list of supported files in the folder
if folder_path:
    file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.splitext(file)[1] in ['.txt', '.pdf', '.docx']]
    st.write(f"Found {len(file_paths)} supported files in the folder.")
else:
    file_paths = []

# Query text
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', disabled=not folder_path)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not(folder_path and query_text))

    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(file_paths, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)