import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
HF_TOKEN = os.getenv("HF_TOKEN") 
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"


def load_llm(repo_id, token):
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=0.5,
        model_kwargs={"token": token, "max_length": "512"}
    )
    return llm

CUSTOM_PROMPT_TEMPLATE = """
Use the conversation history and the provided context to answer the user's question.
If you don’t know the answer, just say that you don’t know, don't make up an answer.
Don't provide anything outside of the given context.

Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["history", "context", "question"])

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"

# Force CPU usage by setting CUDA_VISIBLE_DEVICES to an empty string
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the FAISS vector store
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create a memory object to store conversation history
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Create a conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    memory=memory,
    combine_docs_chain_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Interactive loop for continuous conversation
while True:
    user_query = input("Write Query Here (or type 'exit' to stop): ")
    if user_query.lower() == 'exit':
        break
    
    response = qa_chain.invoke({'question': user_query})
    print("RESULT: ", response["answer"])
