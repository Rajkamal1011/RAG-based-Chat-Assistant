import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, DPRContextEncoder, DPRQuestionEncoder, AutoModelForCausalLM
import torch
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers.bm25 import BM25Retriever
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.docstore import InMemoryDocstore
import faiss
import numpy as np

import torch
# torch.cuda.empty_cache()

import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


# Set device to GPU if available
device = torch.device('cuda:0')
# device = torch.device('cpu')

# Initialize the model checkpoint for the base LLM
checkpoint = "google/gemma-2-2b-it"
print(f"Checkpoint path: {checkpoint}")

# Load tokenizer and base model (LLM for generating responses)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    device_map=device,
    torch_dtype=torch.float32
)

torch.cuda.empty_cache()

# Initialize the LLM pipeline
def llm_pipeline():
    pipe = pipeline(
        'text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=512,
        do_sample=True,
        temperature=0.2,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

def data_ingestion(persist_directory):
    for root, dirs, files in os.walk("docs"):  # Assuming the student handbook is in the 'docs' directory
        for file in files:
            if file.endswith(".pdf"):
                print(f"Processing PDF: {file}")
                loader = PDFMinerLoader(os.path.join(root, file))
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                texts = text_splitter.split_documents(documents)

                # Extract the text content from the Document objects
                text_contents = [doc.page_content for doc in texts]
                
                # Create embeddings
                print("Generating embeddings using all-mpnet-base-v2...")
                model_kwargs = {'device': 'cuda:0'}  # specify GPU device
                embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs=model_kwargs)

                # Initialize FAISS index and add the embeddings
                text_embeddings = embeddings.embed_documents(text_contents)  # Embed text contents
                text_embeddings = np.array(text_embeddings).astype(np.float32)  # Ensure correct format

                index = faiss.IndexFlatL2(len(text_embeddings[0]))
                index.add(text_embeddings)

                # Create the docstore
                docstore = InMemoryDocstore({i: doc for i, doc in enumerate(texts)})
                
                # Create vector store with FAISS
                db = FAISS(embedding_function=embeddings, index=index, docstore=docstore, index_to_docstore_id={i: i for i in range(len(texts))})

                # Persist embeddings
                db.save_local(persist_directory)
                print(f"Embeddings persisted in: {persist_directory}")
                db = None  # Clear the vector store to save memory
                return texts

# Manual DPR Implementation
def get_dpr_retriever(texts):
    # Load pre-trained DPR models for question and context encoding
    question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    
    # Tokenizer for the question and context encoder
    tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

    # Function to retrieve relevant contexts based on a query
    def retrieve_dpr(query, texts):
        question_tokens = tokenizer(query, return_tensors='pt', truncation=True, padding=True)
        question_embedding = question_encoder(**question_tokens).pooler_output.detach().cpu().numpy()
        
        # Encode all the text passages
        context_embeddings = []
        for passage in texts:
            context_tokens = tokenizer(passage.page_content, return_tensors='pt', truncation=True, padding=True)
            context_embedding = context_encoder(**context_tokens).pooler_output.detach().cpu().numpy()
            context_embeddings.append(context_embedding)

        # Use FAISS to index and search
        index = faiss.IndexFlatL2(question_embedding.shape[1])  # L2 distance
        index.add(np.array(context_embeddings).squeeze())
        
        # Search top 5 passages
        _, indices = index.search(question_embedding, k=5)
        return [texts[i].page_content for i in indices[0]]

    return retrieve_dpr

from typing import Any
from pydantic import Field
from langchain.schema import BaseRetriever
from langchain.vectorstores import FAISS
from langchain.retrievers.bm25 import BM25Retriever

class HybridRetriever(BaseRetriever):
    bm25_retriever: BM25Retriever = Field(...)
    dpr_retriever: Any = Field(...)  # Since there's no specific type for DPR retriever
    db: FAISS = Field(...)  # FAISS vector store

    def __init__(self, bm25_retriever, dpr_retriever, db):
        self.bm25_retriever = bm25_retriever
        self.dpr_retriever = dpr_retriever
        self.db = db

    def get_relevant_documents(self, query):
        # BM25 for keyword-based retrieval
        bm25_results = self.bm25_retriever.get_relevant_documents(query)
        
        # DPR for semantic retrieval
        dpr_results = self.dpr_retriever(query, texts)
        
        # FAISS for dense retrieval
        dense_results = self.db.similarity_search(query, k=5)  # Search top 5 matches using FAISS
        
        # Combine all retrieval methods
        combined_results = bm25_results + dpr_results + dense_results
        return combined_results




# Function to create the hybrid RetrievalQA system using FAISS and BM25
from langchain.retrievers import BM25Retriever
from langchain.schema import BaseRetriever

def hybrid_retrieval_llm(texts, persist_directory):
    llm = llm_pipeline()

    # Use all-mpnet-base-v2 for embeddings
    model_kwargs = {'device': 'cuda:0'}  # specify GPU device
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs=model_kwargs)

    # Initialize FAISS index for dense retrieval (load from persisted directory)
    db = FAISS.load_local(persist_directory, embeddings)

    # Extract text content from the `texts`
    bm25_documents = [doc.page_content for doc in texts]

    # Initialize BM25 retriever
    bm25_retriever = BM25Retriever.from_texts(bm25_documents)

    # Use the DPR retriever
    dpr_retriever = get_dpr_retriever(texts)

    # Create a hybrid retriever that conforms to the BaseRetriever interface
    hybrid_retriever = HybridRetriever(bm25_retriever, dpr_retriever, db)

    # Set up the RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Adjust this as needed
        retriever=hybrid_retriever,  # Pass the retriever that conforms to BaseRetriever
        return_source_documents=True
    )
    return qa




# Function to process each query and get the answer
def process_answer(instruction, texts, persist_directory):
    qa = hybrid_retrieval_llm(texts, persist_directory)
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer

# Function to read queries from the Excel file and save the responses
def process_excel_queries(file_path, output_file_path, texts, persist_directory):
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    # Assuming the first column contains queries
    queries = df['question']  # Adjust this based on your actual column name

    # Prepare a list to store results
    results = []
    i = 0
    # Process each query and generate answers
    for query in queries:
        print(f"{i} Processing query: {query}")
        answer = process_answer({'query': query}, texts, persist_directory)
        print(f"Answer: {answer}")
        results.append(answer)
        i += 1

    # Add the responses to the dataframe and save to a new Excel file
    df['response'] = results
    df.to_excel(output_file_path, index=False)
    print(f"Responses saved to {output_file_path}")

if __name__ == "__main__":
    # Persistent directory for FAISS
    persist_directory = "db_faiss"
    
    # Ingest data from the PDF and create embeddings
    print("Starting data ingestion...")
    texts = data_ingestion(persist_directory)
    
    # Process queries and save the responses
    print("Processing textual queries from 'textual_TESTSET_FINAL.xlsx'...")
    process_excel_queries("textual_TESTSET_FINAL.xlsx", "textual_responses_SoTA.xlsx", texts, persist_directory)
