import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_option_menu import option_menu
from llm import load_normal_chain
from htr import save_chat_history_json, get_timestamp, load_chat_history_json
from langchain.memory import StreamlitChatMessageHistory
from image import image_process
import yaml
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

header = st.container()

def load_chain(chat_history):
    return load_normal_chain(chat_history)

def clear_input_field():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""

def set_send_input():
    st.session_state.send_input = True
    clear_input_field()

def track_index():
    st.session_state.session_index_tracker = st.session_state.session_key

def save_chat_history():
    # Check if there is data in the history
    if st.session_state.history != []:
        # If currently in a new session ("New_chat")
        if st.session_state.session_key == "New_chat":
            # Create a new session key based on the current time
            st.session_state.new_session_key = get_timestamp() + ".json"
            file_path = os.path.join(config["chat_history_path"], st.session_state.new_session_key)
        else:
            # Use the current session key if not a new session
            file_path = os.path.join(config["chat_history_path"], st.session_state.session_key)
        
        # Save chat history to a JSON file
        save_chat_history_json(st.session_state.history, file_path)

def RAG_HOME():
    st.title("CHAT WITH AI ASSISTANT")
    chat_container = st.container()
    st.sidebar.title("Chat Sessions")

    # Add 'New_chat' to the list of chat sessions
    chat_sessions = ["New_chat"] + os.listdir(config["chat_history_path"])
    print(chat_sessions)

    # Initialize session state variables if they do not already exist
    if "send_input" not in st.session_state:
        st.session_state.session_key = "New_chat"
        st.session_state.send_input = False
        st.session_state.user_question = ""
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "New_chat"

    if st.session_state.session_key == "New_chat" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None
    
    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox("Select a chat session", chat_sessions, key="session_key", index=index, on_change=track_index)

    if st.session_state.session_key != "New_chat":
        # If session_key is not 'New_chat'
        file_path = os.path.join(config["chat_history_path"], st.session_state.session_key)
        st.session_state.history = load_chat_history_json(file_path)
    else:
        # If session_key is 'New_chat', set history to an empty list
        st.session_state.history = []

    chat_history = StreamlitChatMessageHistory(key="history")
    llm_chain = load_chain(chat_history)

    user_input = st.text_input("Write your message here", key="user_input", on_change=set_send_input)

    send_button = st.button("Send", key="send_button")

    if send_button or st.session_state.send_input:
        if st.session_state.user_question != "":
    
            with chat_container:
                st.chat_message("user").write(st.session_state.user_question)
                llm_response = llm_chain.run(st.session_state.user_question)
                st.session_state.user_question = ""

    if chat_history.messages != []:
        with chat_container:
            st.write("Chat History: ")
            for message in chat_history.messages:
                st.chat_message(message.type).write(message.content)

    save_chat_history()

def RAG():
    # This is the first API key input; no need to repeat it in the main function.
    api_key = 'GOOGLE_API_KEY'
    
    def get_pdf_text(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vector_store(text_chunks, api_key):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
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
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

    def user_input(user_question, api_key):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.success(response["output_text"])

    st.header("CHAT WITH PDF 🤖")

    with st.columns(1)[0]:
        pdf_docs = st.file_uploader(label = "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader", type=['pdf'])
        if st.button("Submit & Process", key="process_button"):  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Done")
    
    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question:
        user_input(user_question, api_key)
    
def streamlit_ui():
    if "session_key" not in st.session_state:
        st.session_state.session_key = "New_chat"
        
    with st.sidebar:
        choice = option_menu('Table of contents', ['Home', 'Chat with PDF/RAG', 'Chat with IMAGE'])
    if choice == 'Home':
        RAG_HOME()
    elif choice == 'Chat with PDF/RAG':
        RAG()
    elif choice == 'Chat with IMAGE':
        image_process()

streamlit_ui()
