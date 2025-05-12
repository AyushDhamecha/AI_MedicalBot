from flask import Flask, request, jsonify
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from translate import Translator

app = Flask(__name__)

DB_FAISS_PATH = "vectorstore/db_faiss"

def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, hf_token):
    if not hf_token:
        raise ValueError("HF_TOKEN is not set. Please check your environment variables.")

    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": hf_token, "max_length": "512"}
    )

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    HF_TOKEN = os.environ.get("HF_TOKEN")
    HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

    if not HF_TOKEN:
        return jsonify({"error": "HF_TOKEN is missing"}), 500

    try:
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don’t know the answer, just say that you don’t know. Don't try to make up an answer.

        Context: {context}
        Question: {question}
        """

        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
        )

        response = qa_chain.invoke({'query': user_message})
        answer = response["result"]

        return jsonify({"response": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
