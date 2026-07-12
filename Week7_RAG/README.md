# 📄 Document Question Answering using RAG

An AI-powered Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents, retrieve relevant information using FAISS semantic search, and generate context-aware answers using Google Gemini.

## Live Demo

**Streamlit App:**  
https://week7-rag-sharanya.streamlit.app/

---

## Features

- Upload PDF documents
- Automatic text extraction and chunking
- Semantic search using FAISS
- Google Gemini embeddings
- Context-aware answer generation
- Download generated answers
- Interactive Streamlit interface

---

##  Technology Stack

- Python
- Streamlit
- Google Gemini API
- LangChain
- FAISS
- PyPDF
- NumPy

---

##  Project Structure

```
Week7_RAG/
│── app.py
│── attention.pdf
│── requirements.txt
│── styles.css
│── README.md
│── week7_SriSharanya.ipynb
└── .streamlit/
    └── secrets.toml
```

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/SriSharanya-617/CEI_Internship.git
cd CEI_Internship/Week7_RAG
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.streamlit/secrets.toml` file:

```toml
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
```

Run the application:

```bash
streamlit run app.py
```

---

##  Workflow

1. Upload a PDF document.
2. Extract text from the document.
3. Split the text into chunks.
4. Generate embeddings using Google Gemini.
5. Store embeddings in a FAISS vector database.
6. Retrieve the most relevant document chunks.
7. Generate accurate answers using Google Gemini.

---

##  Note

A valid Google Gemini API key is required to run this application. Configure your API key in `.streamlit/secrets.toml` before running the project.

---

##  Author

**Sri Sharanya**

Celebal Technologies Internship – Week 7 Project