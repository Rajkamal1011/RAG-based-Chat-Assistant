

import random
import torch
import numpy as np 
import pandas as pd 

CUDA_DEVICE = "cuda:2" 

device = CUDA_DEVICE if torch.cuda.is_available() else "cpu" 

# Import texts and embedding df
text_chunks_and_embedding_df = pd.read_csv("handbook_text_chunks_and_embeddings_df_0.csv") 

# Convert embedding column back to np.array (it got converted to string when it got saved to CSV)
text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

# Convert texts and embedding df to list of dicts
pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records") 

# Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)

#LOADING EMBEDDING MODEL (FOR FINDING EMBEDDINGS OF QUERIES)

from sentence_transformers import util, SentenceTransformer

embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", 
                                      device=CUDA_DEVICE) # choose the device to load the model to

#FUNCTION THAT uses the embedding_model for getting query's embedding and then retrieving relevant chunks' embeddings
#We are not using vector db here, the embeddings loaded serve as our small db
from time import perf_counter as timer

def retrieve_relevant_resources(query: str, 
                                embeddings: torch.tensor,
                                model: SentenceTransformer=embedding_model,
                                n_resources_to_return: int=5, #passages we want to consider - not a hard number, something that you should experiment with
                                print_time: bool=True):
    ''' 
    Embed a query with model and returns top k scores and indices from embeddings.
    '''

    #Embed the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    #Get dot product scores on embeddings
    start_time = timer()
    dot_scores = util.dot_score(query_embedding, embeddings)[0] #0th index indicates the scores I think
    end_time = timer()

    if print_time:
        print(f"[INFO] Time taken to get scores on ({len(embeddings)}) embeddings: {end_time-start_time:.5f} seconds.")
    
    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)

    return scores, indices

# Define helper function to print wrapped text 
import textwrap

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

#Explicitly changing the model and quanitzation config since facing the CUDA out of memory error.
use_quantization_config = False
model_id = "google/gemma-2-2b-it"
print(f"use_quantization_config set to: {use_quantization_config}")
print(f"model_id set to: {model_id}")

#MODEL DEFINITION

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available

# 1. Create a quanitzation config
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_compute_dtype = torch.float16)

if(is_flash_attn_2_available() and (torch.cuda.get_device_capability(0)[0]) >= 8):
    attn_implementation = "flash_attention_2"
else:
    attn_implementation = "sdpa" #scaled dot product attention

# model_id = "google/gemma-7b-it" 
model_id = model_id

# 3. Instantiate tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)

# 4. Instantiate the model
llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, 
                                                 torch_dtype=torch.float16,
                                                 quantization_config=quantization_config if use_quantization_config else None,
                                                 low_cpu_mem_usage=False,
                                                 attn_implementation=attn_implementation)

if not use_quantization_config:
    llm_model.to(CUDA_DEVICE)

print(f"Using attention implementation: {attn_implementation}")

#PROMPT FORMATTER

# def prompt_formatter(query: str,
#                      context_items: list[dict]) -> str:
#     context = "- "+"\n- ".join([item["sentence_chunk"] for item in context_items])
#     base_prompt = """Use the following context items to answer the user query:
# {context}
# \nRelevant passages: <extract relevant passages from the context here>
# User query: {query}
# Answer:"""
#     base_prompt = base_prompt.format(context=context, query=query)

#     #Create prompt template for instruction-tuned model
#     dialogue_template = [
#         {
#             "role": "user",
#             "content": base_prompt
#         }
#     ]

#     #Apply the chat template
#     prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
#                                            tokenize=False,
#                                            add_generation_prompt=True)
    
#     return prompt

# def prompt_formatter(query: str,
#                      context_items: list[dict]) -> str:
#     # Format the context with bullet points
#     context = "\n\n".join([f"Context {i+1}: {item['sentence_chunk']}" for i, item in enumerate(context_items)])
    
#     # More explicit instructions to ensure the model focuses on relevant context
#     base_prompt = f"""
# You are an intelligent assistant. Below are some relevant context items extracted from various documents. Your task is to analyze the context and provide a concise and accurate answer to the user's query based strictly on the provided information. Do not include any information outside the context. If the answer is not found, explicitly say "Answer not found in the provided context."

# Context Information:
# {context}

# User Query: {query}

# Your Answer (based solely on the provided context):
# """
    
#     # Tokenize the prompt
#     dialogue_template = [
#         {
#             "role": "user",
#             "content": base_prompt
#         }
#     ]

#     # Apply the chat template
#     prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
#                                            tokenize=False,
#                                            add_generation_prompt=True)
    
    # return prompt

# def prompt_formatter(query: str,
#                      context_items: list[dict]) -> str:
#     # Format the context items
#     context = "\n".join([f"Context {i+1}: {item['sentence_chunk']}" for i, item in enumerate(context_items[:5])])
    
#     # Construct the conversation in the format Gemma expects
#     base_prompt = f"""
# <start_of_turn>user
# I need an answer based on the following context:
# {context}

# User query: {query}<end_of_turn>
# <start_of_turn>model
# """
    
#     return base_prompt


#BEST WORKING PROMPT
# def prompt_formatter(query: str, context_items: list[dict]) -> str:
#     # Only provide the most relevant context (e.g., the top 2 contexts)
#     context = "\n\n".join([item['sentence_chunk'] for item in context_items[:2]])  # Using only top 2 for simplicity
    
#     # Updated prompt format with no explicit mention of context items
#     base_prompt = f"""
# <start_of_turn>user
# Please answer the following question based on the provided information:
# {context}

# Question: {query}<end_of_turn>
# <start_of_turn>model
# """
    
#     return base_prompt

#Experimental
def prompt_formatter(query: str, context_items: list[dict]) -> str:
    # Only provide the most relevant context (e.g., the top 2 contexts)
    context = "\n\n".join([item['sentence_chunk'] for item in context_items[:2]])  # Using only top 2 for simplicity
    
    # Updated prompt format with no explicit mention of context items
    base_prompt = f""" Act like a chat assistant for students. 
<start_of_turn>user
Please answer the following question in details based on solely following contextual handbook information:
{context}

Question: {query}<end_of_turn>
<start_of_turn>model
"""
    
    return base_prompt

# def prompt_formatter(query: str, context_items: list[dict]) -> str:
#     # Use only the most relevant context items (e.g., top 2)
#     context = "\n".join([item['sentence_chunk'] for item in context_items[:2]])  # Top 2 context items
    
#     # Construct the prompt using the required format, but make it clear that the answer should only come from the context
#     base_prompt = f"""
# <start_of_turn>user
# Here is some relevant information:
# {context}

# Please answer the following question based solely on the provided information.

# Question: {query}<end_of_turn>
# <start_of_turn>model
# """
    
#     return base_prompt



def ask(query, 
        temperature=0.2,
        max_new_tokens=512,
        format_answer_text=True, 
        return_answer_only=True):
    
    # Retrieve relevant resources
    scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings)
    
    # Create a list of context items
    context_items = [pages_and_chunks[i] for i in indices[:5]]  # Use top 5, but you can adjust
    
    # Generate the prompt using the updated format
    prompt = prompt_formatter(query=query, context_items=context_items)
    
    print("Generated Prompt:")
    print(prompt)  # Optional debugging print
    
    # Tokenize the input
    input_ids = tokenizer(prompt, return_tensors="pt").to(CUDA_DEVICE)

    # Generate the model output
    outputs = llm_model.generate(**input_ids, temperature=temperature, do_sample=True, max_new_tokens=max_new_tokens)
    
    # Convert the generated tokens back into text
    output_text = tokenizer.decode(outputs[0])

    if format_answer_text:
        # Remove any residual tags and redundant context text
        output_text = output_text.replace("<start_of_turn>model", "").replace("<end_of_turn>", "").strip()
        
        # Optional: Remove question if model repeats it
        if query in output_text:
            output_text = output_text.replace(query, "").strip()
        
        # Remove everything before and including "Question:" to get the model's answer
        if "Question:" in output_text:
            output_text = output_text.split("Question:")[-1].strip()
        
        # Clean up any leftover boilerplate text (e.g., <bos>, <eos>, etc.)
        output_text = output_text.replace("<bos>", "").replace("<eos>", "").strip()

    return output_text, context_items


query = input("Please Enter your query: ")
output, context_items = ask(query)
print_wrapped(output)
for item in context_items:
    print(item["page_number"])
    print(item["sentence_chunk"])
