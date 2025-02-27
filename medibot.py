import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from translate import Translator

## Uncomment the following files if you're not using pipenv as your virtual environment manager
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

def main():
    st.title("Ask Chatbot!")

    # Add a language selection dropdown
    language_options = {
        "English": "en",
        "Hindi": "hi",
        "Telugu": "te",
        "Tamil": "ta",
        "Sindhi": "sd",
        # Add more languages as needed
    }
    selected_language = st.selectbox("Select Language for Answer", list(language_options.keys()))

    # Initialize session state for messages and sources
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'sources' not in st.session_state:
        st.session_state.sources = []

    # Display previous messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # Display previous sources
    if st.session_state.sources:
        st.subheader("Sources:")
        for source in st.session_state.sources:
            st.markdown(f"**Source:** {source.get('source', 'Unknown')}")
            st.markdown(f"**Page:** {source.get('page', 'Unknown')}")
            st.markdown(f"**Content:** {source.get('content', 'Unknown')}")
            st.markdown("---")

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
        
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try: 
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_documents = response["source_documents"]

            # Translate the result to the selected language
            translator = Translator(to_lang=language_options[selected_language])
            translated_result = translator.translate(result)

            # Display the translated result
            st.chat_message('assistant').markdown(translated_result)
            st.session_state.messages.append({'role': 'assistant', 'content': translated_result})

            # Store the source documents in session state
            for doc in source_documents:
                st.session_state.sources.append({
                    'source': doc.metadata.get('source', 'Unknown'),
                    'page': doc.metadata.get('page', 'Unknown'),
                    'content': doc.page_content
                })

            # Display the source documents in a more readable format
            if st.session_state.sources:
                st.subheader("Sources:")
                for source in st.session_state.sources:
                    st.markdown(f"**Source:** {source.get('source', 'Unknown')}")
                    st.markdown(f"**Page:** {source.get('page', 'Unknown')}")
                    st.markdown(f"**Content:** {source.get('content', 'Unknown')}")
                    st.markdown("---")

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()