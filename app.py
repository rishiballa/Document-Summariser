import streamlit as st
import tempfile
import os
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# --- CONFIGURATION ---
# We use the 1B model because it's under 1GB and fast.
MODEL_NAME = "llama3.2:1b" 

# --- PAGE SETUP ---
st.set_page_config(page_title="Free AI Summarizer", page_icon="📄")
st.title("📄 Private Document Summarizer")
st.caption(f"Powered by Local AI ({MODEL_NAME}) - No Internet Required")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("Settings")
    chunk_size = st.slider("Chunk Size", 500, 3000, 2000, help="Smaller chunks = more detailed but slower.")
    summary_type = st.selectbox("Summary Type", ["map_reduce", "stuff"], index=0, 
                                help="'stuff' is for short docs, 'map_reduce' for long ones.")

# --- MAIN LOGIC ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # 1. Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    if st.button("Generate Summary"):
        status_text = st.empty()
        status_text.info("⏳ Loading AI model and reading file...")

        try:
            # 2. Initialize the Local AI
            llm = OllamaLLM(model=MODEL_NAME)

            # 3. Load PDF
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # 4. Split Text (AI has a memory limit, so we chop the PDF)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=100
            )
            split_docs = text_splitter.split_documents(docs)
            
            status_text.info(f"⏳ Processing {len(split_docs)} chunks of text...")

            # 5. Run Summarization
            chain = load_summarize_chain(llm, chain_type=summary_type)
            summary = chain.invoke(split_docs)

            # 6. Show Result
            status_text.success("✅ Done!")
            st.markdown("### 📝 Summary")
            st.write(summary['output_text'])

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.info("Tip: If the error is 'context length', try reducing Chunk Size in the sidebar.")
            
    # Cleanup temp file

    os.remove(tmp_path)
