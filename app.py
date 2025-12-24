import streamlit as st
from dotenv import load_dotenv
import os 
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class AppUI:
    def __init__(self):
        st.set_page_config(page_title='PDF Chatbot',page_icon=':books:')
        st.title("🤖 DocIntelli: PDF Chatbot")

    def render_sidebar(self):
        with st.sidebar:
            st.header('📁 Document Center')
            uploaded_file = st.file_uploader(
                'Upload your PDF here and click on "Process"',
                type='pdf',
                help='Smaller PDFs process faster'
            )
            process_button = st.button("Process", type="primary", use_container_width=True)

            st.markdown('---')

            if st.button("Clear Conversation"):
                st.session_state.messages = []
                st.rerun()
        return uploaded_file, process_button
    def display_chat_history(self):
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def get_user_input(self):
        # Display the chat input box and returns the prompt
        return st.chat_input("Ask a question about your PDF...")

    def show_error(self, message):
        st.error(message)

    def show_success(self, message):
        st.success(message)

    def show_loading(self, message):
        return st.spinner(message)
    
class DocumentManager:
    def __init__(self,chunk_size=1000,chunk_overlap=200):
        self.splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def get_text_from_pdf(self, uploaded_file):
        reader = PdfReader(uploaded_file)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text()
        
        # Split the giant string into a list of smaller strings
        return self.splitter.split_text(full_text)

class VectorEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None 
    def create_brain(self, text_chunks):
        # FAISS builds the 'search index' using chunks and the local model
        self.vector_store = FAISS.from_texts(
            texts=text_chunks, 
            embedding=self.embeddings
        )
        return self.vector_store
    
    def get_retriever(self):
        if self.vector_store:
            return self.vector_store.as_retriever(search_kwargs={"k": 3})
        return None
    
class ChatNavigator:
    def __init__(self, api_key):
        # Using Gemini 2.5 Flash - the standard speed/intelligence model for 2025
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=api_key,
            temperature=0.2  # Lower temperature = more factual, less creative
        )

    def get_answer(self, question, vector_store):
        """
        Connects the 'Brain' to the 'Voice'.
        1. Finds context 
        2. Fills the prompt 
        3. Gets LLM response
        """
        # 1. Define the 'Retriever' (the search tool)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # 2. Create the System Prompt
        template = """
        You are a helpful AI assistant. Use the following PDF context to answer the question.
        If the answer isn't in the context, say 'I cannot find that in the document.'
        
        Context: {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # 3. Build the Modern Chain (LCEL Syntax)
        # This pipes: Context -> Prompt -> LLM -> Text Output
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain.invoke(question)

if __name__ == "__main__":

    # Initialize OOP Objects
    ui = AppUI()
    doc_manager = DocumentManager()
    vector_engine = VectorEngine()
    
    load_dotenv()
    # Load API Key from environment
    api_key = os.getenv("GOOGLE_API_KEY")

    chat_nav = ChatNavigator(api_key=api_key) if api_key else None

    uploaded_file, process_clicked = ui.render_sidebar()

    # Processing Logic: Triggered only by the 'Process' button
    if uploaded_file and process_clicked:
        with ui.show_loading("Step 1: Extracting text from PDF..."):
            
            chunks = doc_manager.get_text_from_pdf(uploaded_file)
            
        with ui.show_loading("Step 2: Building local vector brain..."):
            # Use VectorEngine to build the FAISS index
            st.session_state.vector_store = vector_engine.create_brain(chunks)
            ui.show_success("Analysis Complete! You can now ask questions.")

    ui.display_chat_history()

    # Get User Question
    if prompt := ui.get_user_input():
        if not api_key:
            ui.show_error("Please add GOOGLE_API_KEY to your .env file.")
        elif "vector_store" not in st.session_state:
            ui.show_error("Please upload and 'Process' a PDF first!")
        else:
            # Add user message to session and display
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate Assistant Answer
            with st.chat_message("assistant"):
                with ui.show_loading("Consulting PDF..."):
                    try:
                        answer = chat_nav.get_answer(prompt, st.session_state.vector_store)
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        ui.show_error(f"Error: {str(e)}")