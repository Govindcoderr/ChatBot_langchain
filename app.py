import gradio as gr
import ollama
import chromadb
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os


# -------------------- Text Extraction -------------------- #
def extract_text(file_paths):
    """Extract text from multiple files (PDF, DOCX, TXT)."""
    text = ""
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    for path in file_paths:
        if path is None:
            continue

        # Handle both path and file-like object
        file_path = path.name if hasattr(path, "name") else path

        try:
            if file_path.lower().endswith(".pdf"):
                pdf_reader = PdfReader(file_path)
                for page in pdf_reader.pages:
                    text += (page.extract_text() or "") + "\n"

            elif file_path.lower().endswith(".docx"):
                doc = Document(file_path)
                for para in doc.paragraphs:
                    text += para.text + "\n"

            elif file_path.lower().endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text += f.read() + "\n"

        except Exception as e:
            return f"⚠️ Error reading {os.path.basename(file_path)}: {e}"

    return text.strip()


# -------------------- Text Chunking -------------------- #
def get_text_chunks(text):
    """Split long text into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)


# -------------------- Vector Store (ChromaDB) -------------------- #
def get_vector_store(text_chunks):
    """Store embeddings of text chunks in ChromaDB."""
    if not text_chunks:
        return None, "❌ No valid text chunks found."

    os.makedirs("chroma_db", exist_ok=True)
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(name="document_embeddings")

    # Clear existing entries (avoid duplicates)
    try:
        collection.delete(where={})
    except Exception:
        pass

    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = embeddings_model.embed_documents(text_chunks)

    metadatas = [{"source": f"chunk_{i}"} for i in range(len(text_chunks))]
    ids = [str(i) for i in range(len(text_chunks))]

    collection.add(
        documents=text_chunks,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings,
    )

    return collection, f"✅ Vector store created with {len(text_chunks)} chunks."

# -------------------- Ollama LLM Integration -------------------- #
def generate_answer(context, question):
    """Generate clean text from Ollama LLM."""
    if not context.strip():
        return "⚠️ No relevant answer found."

    prompt = f"""Answer the following question based on the provided context.

Context:
{context}

Question: {question}
Answer:"""

    try:
        res = ollama.generate(model="llama3.2:1b", prompt=prompt)
        if isinstance(res, tuple):
            res = res[0]
        return res.response.strip()
    except Exception as e:
        return f"⚠️ Ollama LLM error: {e}"

# -------------------- RAG Ask Function -------------------- #
def ask_question(user_question):
    """Retrieve context and generate an answer for the user's query."""
    if not os.path.exists("chroma_db"):
        return "⚠️ Please upload and process documents first."

    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(name="document_embeddings")

    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    query_embedding = embeddings_model.embed_query(user_question)

    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    if not results.get("documents") or not results["documents"][0]:
        return "⚠️ No relevant chunks found for your query."

    context = "\n".join([doc for doc in results["documents"][0] if doc])
    return generate_answer(context, user_question)


# -------------------- File Processing -------------------- #
def process_files(files):
    """Process uploaded files: extract text, split, embed, and store."""
    if not files:
        return "⚠️ Please upload at least one document."

    raw_text = extract_text(files)
    if not raw_text.strip():
        return "⚠️ Could not extract text from the uploaded files."

    chunks = get_text_chunks(raw_text)
    _, msg = get_vector_store(chunks)
    return msg


# -------------------- Gradio UI -------------------- #
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <h2 style='text-align:center'>💬 Multi-File Chatbot (PDF • Word • TXT)</h2>
        <p style='text-align:center;color:gray;'>Upload files and ask questions — powered by Ollama + ChromaDB</p>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(
                label="📂 Upload Documents",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".txt"],
                type="filepath",
            )
            process_btn = gr.Button("⚙️ Process Files", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            chatbox = gr.Chatbot(label="Chat with Your Documents", height=480)
            user_input_txt = gr.Textbox(
                placeholder="Ask something from your uploaded files..."
            )
            send_btn = gr.Button("🚀 Ask", variant="primary")

    process_btn.click(fn=process_files, inputs=file_upload, outputs=status)

    def respond(user_msg, chat_history):
        if not user_msg.strip():
            return chat_history, ""
        answer = ask_question(user_msg)
        chat_history.append((user_msg, answer))
        return chat_history, ""

    send_btn.click(
        fn=respond, inputs=[user_input_txt, chatbox], outputs=[chatbox, user_input_txt]
    )

demo.launch(debug=True , share=True)






