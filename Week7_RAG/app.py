import os
import time
import tempfile
import warnings

import faiss
import numpy as np
import streamlit as st

from google import genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")


# ---------- PAGE CONFIG ----------

st.set_page_config(
    page_title="Document Question Answering using RAG",
    page_icon="📄",
    layout="wide"
)


# ---------- LOAD CSS ----------

def load_css():
    if os.path.exists("styles.css"):
        with open("styles.css") as css:
            st.markdown(
                f"<style>{css.read()}</style>",
                unsafe_allow_html=True
            )


load_css()


# ---------- TITLE ----------

st.title("📄 Document Question Answering using RAG")


st.markdown("""
<div class="hero-card">

<h3>AI Powered Document Intelligence</h3>

<p>
Upload any PDF document and ask natural language questions.
The application retrieves the most relevant document sections using
FAISS semantic search and generates grounded responses using
Google Gemini.
</p>

</div>
""", unsafe_allow_html=True)


# ---------- SIDEBAR ----------

with st.sidebar:

    st.header("Technology Stack")

    st.write("✅ LangChain")
    st.write("✅ Google Gemini")
    st.write("✅ FAISS")
    st.write("✅ Streamlit")

    st.divider()

    st.subheader("Pipeline")

    st.write("""
📄 PDF

⬇

✂ Text Chunking

⬇

🧠 Gemini Embeddings

⬇

📦 FAISS

⬇

🔎 Retrieval

⬇

🤖 Gemini Response
""")


# ---------- API ----------

if "GOOGLE_API_KEY" not in st.secrets:

    st.error("Google API Key not found in Streamlit Secrets.")

    st.stop()

api_key = st.secrets["GOOGLE_API_KEY"]

client = genai.Client(
    api_key=api_key
)

# ---------- PDF ----------

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


# ---------- PROMPT ----------

def create_prompt(question, docs):

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return f"""
You are an intelligent document assistant.

Answer ONLY from the provided context.

If the answer cannot be found, reply:

I could not find the answer in the provided document.

Context:
{context}

Question:
{question}

Answer:
"""


# ---------- BUILD VECTOR DATABASE ----------

@st.cache_resource(show_spinner=False)
def build_vector_database(pdf_path):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(
        documents
    )

    embeddings = []

    progress = st.progress(0)

    status = st.empty()

    for i, doc in enumerate(chunks):

        for attempt in range(3):

            try:
                response = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=doc.page_content
                )

                embeddings.append(
                    response.embeddings[0].values
                )

                break

            except Exception:

                if attempt == 2:
                    raise

                time.sleep(15)

        progress.progress(
            (i + 1) / len(chunks)
        )

        status.text(
            f"Generating Embeddings : {i + 1}/{len(chunks)}"
        )

    embedding_matrix = np.array(
        embeddings,
        dtype=np.float32
    )

    dimension = embedding_matrix.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embedding_matrix
    )

    progress.empty()

    status.empty()

    return (
        documents,
        chunks,
        index,
        dimension
    )
    # ---------- PROCESS PDF ----------

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".pdf"
    ) as temp_file:

        temp_file.write(
            uploaded_file.read()
        )

        pdf_path = temp_file.name

    st.success(f"Uploaded File: {uploaded_file.name}")
    start_time = time.time()

    with st.spinner("Building Vector Database..."):

        documents, chunks, index, dimension = build_vector_database(
            pdf_path
        )

    end_time = time.time()

    st.success("Vector Database created successfully.")

    st.info(
        f"Processing Time : {end_time-start_time:.2f} seconds"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Pages",
            len(documents)
        )

    with col2:

        st.metric(
            "Chunks",
            len(chunks)
        )

    with col3:

        st.metric(
            "Embedding",
            dimension
        )

    with col4:

        st.metric(
            "Top-K",
            3
        )


# ---------- RETRIEVER ----------

    def retrieve(question, k=3):

        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=question
        )

        query_embedding = np.array(
            [response.embeddings[0].values],
            dtype=np.float32
        )

        distances, indices = index.search(
            query_embedding,
            k
        )

        retrieved_docs = []

        for idx in indices[0]:

            retrieved_docs.append(
                chunks[idx]
            )

        return retrieved_docs


# ---------- QUESTION ----------

    st.markdown("## Ask a Question")

    question = st.text_input(
        "Enter your question"
    )

    ask = st.button(
        "Generate Answer"
    )


# ---------- ANSWER ----------

    if ask:

        if question.strip() == "":

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching relevant information..."
            ):

                retrieved_docs = retrieve(
                    question
                )

                prompt = create_prompt(
                    question,
                    retrieved_docs
                )

                try:

                    response = client.models.generate_content(
                        model="gemini-3.5-flash",
                        contents=prompt
                    )

                    answer = response.text

                except Exception as e:

                    st.error(
                        "Unable to generate response."
                    )

                    st.exception(e)

                    st.stop()
            st.subheader("Generated Answer")

            st.markdown(
                f"""
                <div class="answer-box">
                    {answer}
                </div>
                """,
                unsafe_allow_html=True
            )
                        
            st.success("Response generated successfully.")

            with st.expander("Retrieved Document Chunks", expanded=False):

                for i, doc in enumerate(retrieved_docs, start=1):

                    st.markdown(f"### Chunk {i}")

                    st.write(doc.page_content)

                    st.divider()


            st.subheader("System Summary")

            summary_col1, summary_col2 = st.columns(2)

            with summary_col1:

                st.write(f"**Pages Loaded:** {len(documents)}")
                st.write(f"**Document Chunks:** {len(chunks)}")
                st.write(f"**Embedding Dimension:** {dimension}")
                st.write(f"**Uploaded File:** {uploaded_file.name}")

            with summary_col2:

                st.write("**Embedding Model:** Gemini Embedding 001")
                st.write("**Vector Database:** FAISS")
                st.write("**Language Model:** Gemini 3.5 Flash")
                st.write(f"**Retrieved Chunks:** {len(retrieved_docs)}")


            st.download_button(
                label="Download Answer",
                data=answer,
                file_name="rag_answer.txt",
                mime="text/plain"
            )


    if os.path.exists(pdf_path):

        os.remove(pdf_path)

else:

    st.info(
        "Upload a PDF document to begin asking questions."
    )


st.markdown("---")

st.markdown("""
<div style="text-align:center;color:#666;font-size:14px;">

Document Question Answering using RAG<br>

Powered by Streamlit • LangChain • Google Gemini • FAISS

</div>
""", unsafe_allow_html=True)