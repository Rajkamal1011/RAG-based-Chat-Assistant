import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS

# Set device to GPU (or change to CPU if needed)
device = torch.device('cuda:2')

# Initialize the local model path where gemma-2-2b-it is stored
local_model_path = "./gemma-2-2b-it"
print(f"Local Model path: {local_model_path}")

# Load the tokenizer and base model from the local directory
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    device_map=device,
    torch_dtype=torch.float32
)

# Set the persistent directory for embeddings
persist_directory = "db"

# Function to ingest PDF data
def data_ingestion():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2", device="cuda:2")
    # Create vector store
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

# Initialize the LLM pipeline
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        device=device
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

# Function to create the RAG-based RetrievalQA system
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2", device="cuda:2")
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

# Function to process each query and get the answer
def process_answer(instruction):
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer

# Function to read queries from the Excel file and save the responses
def process_excel_queries(file_path, output_file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    # Assuming the first column contains queries
    queries = df['question']  # Adjust this based on your actual column name

    # Prepare a list to store results
    results = []

    # Process each query and generate answers
    for query in queries:
        print(f"Processing query: {query}")
        answer = process_answer({'query': query})
        print(f"Answer: {answer}")
        results.append(answer)

    # Add the responses to the dataframe and save to a new Excel file
    df['response'] = results
    df.to_excel(output_file_path, index=False)
    print(f"Responses saved to {output_file_path}")

if __name__ == "__main__":
    # First, we ingest the data and build the embeddings (skip this if embeddings already exist)
    data_ingestion()

    # Process textual queries and save responses to 'abstr_textual_TESTSET_RESPONSES.xlsx'
    print("Processing textual queries from 'textual_TESTSET_FINAL.xlsx'...")
    process_excel_queries("textual_TESTSET_FINAL.xlsx", "gemma_abstr_textual_TESTSET_RESPONSES.xlsx")

    # Process tabular queries and save responses to 'abstr_tabular_TESTSET_RESPONSES.xlsx'
    print("Processing tabular queries from 'tabular_TESTSET_FINAL.xlsx'...")
    process_excel_queries("tabular_TESTSET_FINAL.xlsx", "gemma_abstr_tabular_TESTSET_RESPONSES.xlsx")

