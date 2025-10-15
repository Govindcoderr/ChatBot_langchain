     ## 💬 Multi-File Chatbot (PDF • DOCX • TXT)

An intelligent **document-based chatbot** that allows users to upload multiple files (PDF, Word, or Text) and ask questions based on their content.
Built with **Gradio**, **LangChain**, **Ollama**, and **ChromaDB**, it performs **retrieval-augmented generation (RAG)** for accurate, context-aware answers.

---

### 🚀 Features

* 📂 Supports **multiple document formats** — PDF, DOCX, TXT
* 🧠 Uses **LangChain** for text chunking and embedding
* ⚙️ **ChromaDB** for efficient vector storage and similarity search
* 🤖 **Ollama LLM integration** (local model like `llama3.2:1b`)
* 🪶 Simple and elegant **Gradio UI**
* 🔍 Ask questions directly from your uploaded files
* 💡 Fast, accurate, and self-contained (no external APIs)

---

### 🧩 Tech Stack

| Layer           | Technology                       |
| --------------- | -------------------------------- |
| Frontend        | Gradio                           |
| Backend         | Python                           |
| LLM             | Ollama (`llama3.2:1b`)           |
| Embeddings      | HuggingFace (`all-MiniLM-L6-v2`) |
| Vector Database | ChromaDB                         |
| File Parsing    | PyPDF2, python-docx              |

---

### ⚙️ Installation & Setup

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Govindcoderr/ChatBot_langchain.git
cd ChatBot_langchain
```

#### 2️⃣ Create and Activate Virtual Environment

```bash
python -m venv .venv
# Activate on Windows
.venv\Scripts\activate
# or on Linux/Mac
source .venv/bin/activate
```

#### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4️⃣ Start Ollama (make sure Ollama is running)

Download and run [Ollama](https://ollama.ai/) locally, then pull the model:

```bash
ollama pull llama3.2:1b
```

#### 5️⃣ Run the Application

```bash
python app.py
```

#### 6️⃣ Access the App

Visit:

```
http://127.0.0.1:7860
```

Or use the public Gradio link (if `share=True` is enabled).

---

### 🗂️ Project Structure

```
ChatBot_langchain/
│
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── .gitignore             # Ignored files
├── chroma_db/             # Local vector database
└── README.md              # Project documentation
```

---

### 🧠 How It Works

1. **Upload documents** → PDF, DOCX, or TXT
2. **Extract text** using PyPDF2 / python-docx
3. **Split into chunks** using LangChain’s RecursiveCharacterTextSplitter
4. **Generate embeddings** with HuggingFace
5. **Store vectors** in ChromaDB
6. **Query user question** → find top relevant chunks
7. **Ollama LLM** generates the final, context-aware response

---

### 🧰 Requirements

See [`requirements.txt`](./requirements.txt)

---


### 🧑‍💻 Author

**Govind Rajpurohit**
💼 Software Developer | Gen Ai Developer 
📍 India

🔗 GitHub: [@Govindcoderr](https://github.com/Govindcoderr)

---

### 🏷️ License

This project is licensed under the **MIT License** — you’re free to modify and use it for personal or commercial projects.

---

<img width="2857" height="1521" alt="image" src="https://github.com/user-attachments/assets/f2ce0008-126d-4fc7-b747-29ad16c7f927" />

