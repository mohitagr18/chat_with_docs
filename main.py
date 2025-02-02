# Import libraries
import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from a .env file
load_dotenv()

# Configure the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Read pdf and extract text
def read_pdf(pdf_docs):
    """
    Reads text from a list of PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text and return chunks
def get_text_chunks(text):
    """
    Splits the given text into chunks of specified size with overlap.
    """
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks 

# Create vector store
def get_vector_store(text_chunks):
    """
    Generates a vector store from given text chunks using Google Generative AI embeddings and FAISS.
    1. Creates embeddings for the text chunks using GoogleGenerativeAIEmbeddings.
    2. Creates a FAISS vector store from the embeddings.
    3. Saves the vector store locally as "faiss_index".
    """
    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    # Create vector store
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    # Save vector store
    vector_store.save_local("faiss_index")
    return vector_store

# Load conversational chain
def get_conversational_chain():
    """
    Creates and returns a conversational chain for answering questions based on provided context.
    """
    prompt_template = '''
    Answer the question as detailed as possible from the provided context and make sure to provide
    all details. If the answer is not in the context, just say "answer is not in the context".
    Do not provide an incorrect answer.
    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
    '''

    # Load model and chain
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, 
                            input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Generate response
def user_input(user_question):
    """
    Processes the user's question by searching for similar documents and generating a response.
        1. Load the vector store using GoogleGenerativeAIEmbeddings.
        2. Perform a similarity search on the vector store with the user's question.
        3. Retrieve the conversational chain.
        4. Generate a response based on the similar documents and the user's question.
        5. Print and display the response using Streamlit.
    """
    # Load vector store
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # Search for similar documents
    docs = new_db.similarity_search(user_question)
    # Get conversational chain
    chain = get_conversational_chain()
    # Get response
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    print(response)
    st.subheader("The Response is:")
    st.write(response["output_text"])

# Initialize session state
def initialize_session_state():
    """
    Initializes the session state variables for the Streamlit application.
    - pdf_processed (bool): Indicates whether the PDF has been processed. Default is False.
    - query_count (int): Tracks the number of queries made. Default is 0.
    """
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "query_count" not in st.session_state:
        st.session_state['query_count'] = 0

# Manage query count
def manage_query_count():
    """
    Manages the query count for the session.
    If the query count exceeds the limit of 5, a warning message is displayed.
    """
    if st.session_state['query_count'] > 5:
        st.warning("You have reached the limit of 5 queries. Please check back later.")
        return

# Handle user question
def handle_user_question(user_question):
    """
    Handles the user's question by checking if the PDF documents have been processed.
    If the PDF documents have been processed, it increments the query count and processes the user input.
    Otherwise, it displays a warning message to upload and process PDF documents first.
    """
    if st.session_state.pdf_processed:
        st.session_state['query_count'] += 1
        user_input(user_question)
    else:
        st.warning("Please upload and process PDF documents first.")

# Handle PDF upload
def handle_pdf_upload(pdf_docs):
    """
    Handles the upload and processing of PDF documents.
    The function performs the following steps:
    1. Checks if any PDF documents are uploaded.
    2. If documents are uploaded, displays a spinner while processing.
    3. Reads the PDF documents and extracts text.
    4. Splits the extracted text into chunks.
    5. Creates a vector store from the text chunks.
    6. Sets the session state to indicate that the PDF processing is complete.
    7. Displays a success message upon completion.
    8. If no documents are uploaded, displays a warning message.
    """
    # Process PDF documents
    if pdf_docs:
        with st.spinner("Processing..."):
            # Read PDF and extract text
            raw_text = read_pdf(pdf_docs)
            # Split text into chunks
            text_chunks = get_text_chunks(raw_text)
            # Create vector store
            get_vector_store(text_chunks)
            # Set session state
            st.session_state.pdf_processed = True
            st.success("Processing complete!")
    else:
        st.warning("Please upload at least one PDF document.")

# Define the main function
def main():
    """
    Main function to set up the Streamlit app interface for interacting with PDF documents using Google Gemini.
    1. Sets the page configuration and header.
    2. Displays a markdown description of the app.
    3. Initializes session state and manages query count.
    4. Provides a text input for the user to ask questions based on the uploaded PDF documents.
    5. Handles the user's question upon submission.
    6. Provides a sidebar menu for uploading PDF documents and processing them.
    The app allows users to upload one or more PDF files and ask questions to receive detailed responses based on the content of the uploaded documents.
    """
    # Set page configuration
    st.set_page_config(page_title="AI PDF Explorer", page_icon=":books:", layout="wide")
    
    # Set header and description
    st.header("AI-Powered PDF Explorer")
    st.markdown("---")
    st.markdown("###### Unlock the knowledge hidden in your PDFs. This powerful web app uses Google Gemini to analyze and answer your questions about the uploaded documents, saving you time and effort. Perfect for quickly finding information, summarizing reports, and gaining deeper understanding.")
    
    # Initialize session state and manage query count
    initialize_session_state()
    manage_query_count()

    # User input for questions
    user_question = st.text_input("Ask a question from the PDF files uploaded:")
    if st.button("Submit"):
        handle_user_question(user_question)

    # Sidebar for PDF upload
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
        
        if st.button("Submit and Process"):
            handle_pdf_upload(pdf_docs)



if __name__ == "__main__":
    main()  # Call the main function