from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline

# Step 1: Load the pretrained Sentence Transformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Load the PDF document and extract text
pdfreader = PdfReader('student_information_handbook.pdf')

raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# Step 3: Split the text into manageable chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# Print the number of text chunks
print(f"Number of text chunks: {len(texts)}")

# Step 4: Generate embeddings for each chunk of text using Sentence Transformer
embeddings = [embedding_model.encode(text) for text in texts]

# Print type and length of embeddings list
print(f"Type of embeddings: {type(embeddings)}")
print(f"Number of embeddings: {len(embeddings)}")

# Print the shape of the first embedding (to ensure they are vectors)
print(f"Shape of the first embedding: {np.array(embeddings[0]).shape}")

# Convert embeddings to a 2D NumPy array
embeddings = np.vstack(embeddings)

# Print shape of embeddings after stacking
print(f"Shape of embeddings (2D array): {embeddings.shape}")

# Step 5: Create a FAISS index with the embeddings and texts
document_search = FAISS.from_embeddings(embeddings, texts)

# Step 6: Load the "google/gemma-2-2b-it" model for question answering
model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Step 7: Create a Hugging Face pipeline for question answering
hf_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Step 8: Initialize the HuggingFacePipeline for LangChain
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Step 9: Load the QA chain using the Hugging Face pipeline
chain = load_qa_chain(llm, chain_type="stuff")

# Step 10: Run queries using the FAISS search and the QA chain
# Example query 1
query = "Vision for Amrit Kaal"
docs = document_search.similarity_search(query)
result = chain.run(input_documents=docs, question=query)
print(f"Query: {query}\nResult: {result}\n")

# Example query 2
query = "How much the agriculture target will be increased to and what the focus will be"
docs = document_search.similarity_search(query)
result = chain.run(input_documents=docs, question=query)
print(f"Query: {query}\nResult: {result}\n")
