import streamlit as st
import os
from src.engine import DocumentManager, RAGEngine
from src.styles import apply_custom_styles

# Page configuration
st.set_page_config(
    page_title="PolicyExpert AI",
    page_icon="🛡️",
    layout="wide"
)

# Apply premium styles
apply_custom_styles()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# Sidebar for configuration
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=80)
    st.title("🛡️ PolicyExpert AI")
    st.markdown("*Intelligent Governance Assistant*")
    st.markdown("---")
    
    provider = st.selectbox("LLM Provider", ["openai", "groq"], help="Select your preferred AI model provider")
    api_key = st.text_input(f"{provider.capitalize()} API Key", type="password", help="Enter your API key to enable the AI")
    
    st.markdown("---")
    st.subheader("📄 Document Center")
    uploaded_files = st.file_uploader("Upload Policy Documents", type=["pdf", "txt"], accept_multiple_files=True)
    
    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("Process New", use_container_width=True)
    with col2:
        sync_btn = st.button("Sync Data", use_container_width=True)

    if (process_btn and uploaded_files) or sync_btn:
        with st.status("Initializing engine...", expanded=True) as status:
            doc_manager = DocumentManager()
            all_chunks = []
            
            # Handle uploaded files
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    file_path = os.path.join("data/uploads", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.write(f"Processing: {uploaded_file.name}")
                    chunks = doc_manager.load_and_split(file_path)
                    all_chunks.extend(chunks)
                    if uploaded_file.name not in st.session_state.processed_files:
                        st.session_state.processed_files.append(uploaded_file.name)
            
            # Handle existing files in data/
            if sync_btn:
                st.write("Syncing existing policies...")
                existing_chunks = doc_manager.load_directory("data")
                all_chunks.extend(existing_chunks)
                st.session_state.processed_files.extend([f for f in os.listdir("data") if f.endswith((".txt", ".pdf"))])

            if all_chunks:
                st.write("Generating neural search index...")
                st.session_state.rag_engine = RAGEngine(api_key=api_key, provider=provider)
                st.session_state.rag_engine.create_vector_store(all_chunks)
                status.update(label="System Ready!", state="complete", expanded=False)
                st.success(f"Indexed {len(st.session_state.processed_files)} documents!")
            else:
                status.update(label="No documents found", state="error", expanded=False)

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Header
st.title("Welcome to PolicyExpert AI")
st.markdown("### Ask anything about your organization's policies, guidelines, and compliance.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message and message["context"]:
            with st.expander("🔍 View Referenced Sources"):
                for i, doc in enumerate(message["context"]):
                    source_name = doc.metadata.get('source', 'Policy Document').split('\\')[-1].split('/')[-1]
                    st.markdown(f"**[{i+1}] {source_name}**")
                    st.info(doc.page_content)

# Chat Input
if prompt := st.chat_input("Ex: What is the company policy on remote work?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if st.session_state.rag_engine:
            with st.spinner("Analyzing cross-referenced documents..."):
                answer, context = st.session_state.rag_engine.get_response(prompt)
                st.markdown(answer)
                
                if context:
                    with st.expander("🔍 View Referenced Sources"):
                        for i, doc in enumerate(context):
                            source_name = doc.metadata.get('source', 'Policy Document').split('\\')[-1].split('/')[-1]
                            st.markdown(f"**[{i+1}] {source_name}**")
                            st.info(doc.page_content)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "context": context
                })
        else:
            st.warning("⚠️ Engine not initialized. Please enter your API key and sync/upload documents in the sidebar.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: rgba(255,255,255,0.4); font-size: 0.8rem;'>"
    "PolicyExpert AI v2.0 | Secured by Enterprise Encryption | Powered by Antigravity AI"
    "</div>", 
    unsafe_allow_html=True
)
