import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

# Load environment variables from .env file
load_dotenv()

# Get Google API key from environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if the GOOGLE_API_KEY is loaded properly
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in the .env file.")

# Configure the API key for Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Set custom background image for the Streamlit page (related to AI technology)
st.markdown(
    """
    <style>
    .main {
        background-image: url('https://images.unsplash.com/photo-1517694712202-14dd9538aa97');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>input {
        background-color: rgba(255, 255, 255, 0.6);
        color: black;
    }
    </style>
    """, unsafe_allow_html=True
)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_txt_text(txt_docs):
    text = ""
    for txt in txt_docs:
        text += txt.getvalue().decode("utf-8")
    return text

def get_website_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        st.error(f"Failed to fetch website content: {e}")
        return ""

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    if "output_text" in response:
        st.write("Reply: ", response["output_text"])
    else:
        st.write("Error: Could not generate a response.")

def main():
    st.header("NPS'S CHATBOT")
    
    user_question = st.text_input("ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಇಲ್ಲಿ ನಮೂದಿಸಿ")
    
    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        # File uploader for PDF and TXT files
        pdf_docs = st.file_uploader("Upload PDF", accept_multiple_files=True, type="pdf")
        txt_docs = st.file_uploader("Upload Text File", accept_multiple_files=True, type="txt")
        
        # URL input for website content
        url = st.text_input("Enter Website URL")
        
        if st.button("Submit & Process"):
            with st.spinner("Processing... 💃"):
                raw_text = ""
                # Process PDF files
                if pdf_docs:
                    raw_text += get_pdf_text(pdf_docs)
                # Process Text files
                if txt_docs:
                    raw_text += get_txt_text(txt_docs)
                # Process Website content
                if url:
                    raw_text += get_website_text(url)
                
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete!")
                else:
                    st.error("No files or URL provided.")

if __name__ == "__main__":
    main()
