import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from translate import Translator

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, hf_token):
    if not hf_token:
        raise ValueError("HF_TOKEN is not set. Please check your environment variables.")
    
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": hf_token, "max_length": "512"}
    )
    return llm

def main():
    st.title("Document Summarization & Conversational Chatbot")

    language_options = {
        "English": "en",
        "Hindi": "hi",
        "Telugu": "te",
        "Tamil": "ta",
        "Sindhi": "sd",
    }
    selected_language = st.selectbox("Select Language for Summary", list(language_options.keys()))

    # Load environment variables
    HF_TOKEN = os.environ.get("HF_TOKEN")
    HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

    if not HF_TOKEN:
        st.error("HF_TOKEN is missing. Set it in your environment variables.")
        return

    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if st.session_state.summary:
        st.subheader("Summary:")
        st.markdown(st.session_state.summary)

    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            try:
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")
                    return

                retriever = vectorstore.as_retriever(search_kwargs={'k': 10})
                docs = retriever.get_relevant_documents("")
                context = "\n".join([doc.page_content for doc in docs])

                CUSTOM_PROMPT_TEMPLATE = """
                You are an expert in summarizing documents. Generate a concise summary following these guidelines:
                1. Focus on key points.
                2. Exclude irrelevant details.
                3. Ensure clarity within 200 words.
                
                Document: {context}
                
                Summary:
                """

                llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({'query': "Summarize the document", 'context': context})
                summary = response["result"]

                translator = Translator(to_lang=language_options[selected_language])
                translated_summary = translator.translate(summary)

                st.session_state.summary = translated_summary
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if st.session_state.summary:
        st.subheader("Ask Questions About the Summary")

        for message in st.session_state.messages:
            st.chat_message(message['role']).markdown(message['content'])

        prompt = st.chat_input("Ask your question here")

        if prompt:
            st.chat_message('user').markdown(prompt)
            st.session_state.messages.append({'role': 'user', 'content': prompt})

            CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you don’t know the answer, just say that you don’t know, don't try to make up an answer.
                Don't provide anything out of the given context.

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
            """

            try:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                    chain_type="stuff",
                    retriever=get_vectorstore().as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=False,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({'query': prompt, 'context': st.session_state.summary})
                answer = response["result"]

                translator = Translator(to_lang=language_options[selected_language])
                translated_answer = translator.translate(answer)

                st.chat_message('assistant').markdown(translated_answer)
                st.session_state.messages.append({'role': 'assistant', 'content': translated_answer})
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
