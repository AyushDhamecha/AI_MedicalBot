# import os
# import streamlit as st

# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint
# from translate import Translator

# DB_FAISS_PATH = "vectorstore/db_faiss"

# @st.cache_resource
# def get_vectorstore():
#     embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db

# def set_custom_prompt(custom_prompt_template):
#     prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt

# def load_llm(huggingface_repo_id, hf_token):
#     if not hf_token:
#         raise ValueError("HF_TOKEN is not set. Please check your environment variables.")
    
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={"token": hf_token, "max_length": "512"}
#     )
#     return llm

# def main():
#     st.title("Document Summarization & Conversational Chatbot")

#     language_options = {
#         "English": "en",
#         "Hindi": "hi",
#         "Telugu": "te",
#         "Tamil": "ta",
#         "Sindhi": "sd",
#     }
#     selected_language = st.selectbox("Select Language for Summary", list(language_options.keys()))

#     # Load environment variables
#     HF_TOKEN = os.environ.get("HF_TOKEN")
#     HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

#     if not HF_TOKEN:
#         st.error("HF_TOKEN is missing. Set it in your environment variables.")
#         return

#     if 'summary' not in st.session_state:
#         st.session_state.summary = None
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

#     if st.session_state.summary:
#         st.subheader("Summary:")
#         st.markdown(st.session_state.summary)

#     if st.button("Generate Summary"):
#         with st.spinner("Generating summary..."):
#             try:
#                 vectorstore = get_vectorstore()
#                 if vectorstore is None:
#                     st.error("Failed to load the vector store")
#                     return

#                 retriever = vectorstore.as_retriever(search_kwargs={'k': 10})
#                 docs = retriever.get_relevant_documents("")
#                 context = "\n".join([doc.page_content for doc in docs])

#                 CUSTOM_PROMPT_TEMPLATE = """
#                 You are an expert in summarizing documents. Generate a concise summary following these guidelines:
#                 1. Focus on key points.
#                 2. Exclude irrelevant details.
#                 3. Ensure clarity within 200 words.
                
#                 Document: {context}
                
#                 Summary:
#                 """

#                 llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)

#                 qa_chain = RetrievalQA.from_chain_type(
#                     llm=llm,
#                     chain_type="stuff",
#                     retriever=retriever,
#                     return_source_documents=True,
#                     chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#                 )

#                 response = qa_chain.invoke({'query': "Summarize the document", 'context': context})
#                 summary = response["result"]

#                 translator = Translator(to_lang=language_options[selected_language])
#                 translated_summary = translator.translate(summary)

#                 st.session_state.summary = translated_summary
#                 st.rerun()
#             except Exception as e:
#                 st.error(f"Error: {str(e)}")

#     if st.session_state.summary:
#         st.subheader("Ask Questions About the Summary")

#         for message in st.session_state.messages:
#             st.chat_message(message['role']).markdown(message['content'])

#         prompt = st.chat_input("Ask your question here")

#         if prompt:
#             st.chat_message('user').markdown(prompt)
#             st.session_state.messages.append({'role': 'user', 'content': prompt})

#             CUSTOM_PROMPT_TEMPLATE = """
#                 Use the pieces of information provided in the context to answer user's question.
#                 If you don’t know the answer, just say that you don’t know, don't try to make up an answer.
#                 Don't provide anything out of the given context.

#                 Context: {context}
#                 Question: {question}

#                 Start the answer directly. No small talk please.
#             """

#             try:
#                 qa_chain = RetrievalQA.from_chain_type(
#                     llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
#                     chain_type="stuff",
#                     retriever=get_vectorstore().as_retriever(search_kwargs={'k': 3}),
#                     return_source_documents=False,
#                     chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#                 )

#                 response = qa_chain.invoke({'query': prompt, 'context': st.session_state.summary})
#                 answer = response["result"]

#                 translator = Translator(to_lang=language_options[selected_language])
#                 translated_answer = translator.translate(answer)

#                 st.chat_message('assistant').markdown(translated_answer)
#                 st.session_state.messages.append({'role': 'assistant', 'content': translated_answer})
#             except Exception as e:
#                 st.error(f"Error: {str(e)}")

# if __name__ == "__main__":
#     main()


import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from translate import Translator
from create_memory_for_llm import process_uploaded_file

def get_vectorstore(db_path):
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, hf_token):
    if not hf_token:
        raise ValueError("HF_TOKEN is not set.")
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": hf_token, "max_length": "512"}
    )

def main():
    st.set_page_config(page_title="MediBot AI - AI Medical Assistant", page_icon="🩺")
    
    st.markdown("""
    <style>
    .main-title { font-size: 36px; font-weight: bold; color: #2E86C1; text-align: center; margin-bottom: 20px; }
    .sub-title { font-size: 20px; color: #5D6D7E; text-align: center; margin-bottom: 30px; }
    .upload-section { margin-bottom: 20px; padding: 20px; background-color: #F8F9F9; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">MediBot AI - AI Medical Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Empowering health decisions with AI</div>', unsafe_allow_html=True)

    # Initialize session state
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'uploaded_docs' not in st.session_state:
        st.session_state.uploaded_docs = None
    if 'db_path' not in st.session_state:
        st.session_state.db_path = None

    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a medical PDF document", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Processing your document..."):
            success, result = process_uploaded_file(uploaded_file)
            if success:
                st.success(result["message"])
                st.session_state.uploaded_docs = result["documents"]
                st.session_state.db_path = result["db_path"]
            else:
                st.error(f"Error: {result}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Language selection
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

# ... (previous imports and code remain the same until the button section)

    # Generate summary from uploaded document - MODIFIED BUTTON SECTION
    if st.session_state.uploaded_docs:
        if st.button("Generate Summary", key="generate_summary_button"):
            with st.spinner("Generating summary..."):
                try:
                    # Get the full text from uploaded documents
                    context = "\n".join([doc.page_content for doc in st.session_state.uploaded_docs])

                    CUSTOM_PROMPT_TEMPLATE = """
                    You are an expert in summarizing medical documents. Generate a concise summary following these guidelines:
                    1. Focus on key medical information.
                    2. Exclude irrelevant details.
                    3. Ensure clarity within 200 words.
                    
                    Document: {context}
                    
                    Summary:
                    """

                    llm = load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN)
                    
                    # Create prompt with the uploaded document's content
                    prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
                    formatted_prompt = prompt.format(context=context, question="Summarize this medical document")

                    # Get summary from LLM
                    response = llm.invoke(formatted_prompt)
                    summary = response

                    # Translate if needed
                    if selected_language != "en":
                        translator = Translator(to_lang=language_options[selected_language])
                        summary = translator.translate(summary)

                    st.session_state.summary = summary
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
    else:
        st.warning("Please upload a PDF document first to generate summary")

    # ... (rest of the code remains the same)

    # Display summary and chat interface
    if st.session_state.summary:
        st.subheader("Document Summary:")
        st.markdown(st.session_state.summary)

        st.subheader("Ask Questions About the Document")
        for message in st.session_state.messages:
            st.chat_message(message['role']).markdown(message['content'])

        prompt = st.chat_input("Ask your question here")
        if prompt and st.session_state.db_path:
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            
            try:
                # Use the session-specific vector store
                db = get_vectorstore(st.session_state.db_path)
                retriever = db.as_retriever(search_kwargs={'k': 3})
                
                CUSTOM_PROMPT_TEMPLATE = """
                Use the medical document to answer the question accurately.
                If you don't know the answer, say you don't know.
                Be precise and professional.

                Context: {context}
                Question: {question}

                Answer:
                """
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({'query': prompt})
                answer = response["result"]

                if selected_language != "en":
                    translator = Translator(to_lang=language_options[selected_language])
                    answer = translator.translate(answer)

                st.session_state.messages.append({'role': 'assistant', 'content': answer})
                st.rerun()
            except Exception as e:
                st.error(f"Error answering question: {str(e)}")

if __name__ == "__main__":
    main()