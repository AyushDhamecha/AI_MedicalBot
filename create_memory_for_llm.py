from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

## Uncomment the following files if you're not using pipenv as your virtual environment manager
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())

# Step 1: Load raw PDF(s)
DATA_PATH = "data/"

def load_pdf_files(data):
    """
    Load PDF files from the specified directory.
    """
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Load PDF documents
documents = load_pdf_files(data=DATA_PATH)
print("Number of PDF pages loaded: ", len(documents))

# Step 2: Create Chunks
def create_chunks(extracted_data):
    """
    Split the extracted text into smaller chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Size of each chunk
        chunk_overlap=50  # Overlap between chunks for context
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Create text chunks
text_chunks = create_chunks(extracted_data=documents)
print("Number of text chunks created: ", len(text_chunks))

# Step 3: Create Vector Embeddings
def get_embedding_model():
    """
    Initialize the embedding model for generating vector representations of text.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

# Initialize the embedding model
embedding_model = get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_faiss_vectorstore(text_chunks, embedding_model):
    """
    Create a FAISS vector store from the text chunks and save it locally.
    """
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print("FAISS vector store created and saved locally at:", DB_FAISS_PATH)

# Create and save the FAISS vector store
create_faiss_vectorstore(text_chunks, embedding_model)