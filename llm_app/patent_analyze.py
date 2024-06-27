import os
import torch
from typing import List
from datasets import load_dataset
from datasets import concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    pipeline,
    BitsAndBytesConfig,
)

from langchain import HuggingFacePipeline
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.prompts import PromptTemplate

from langchain.chains import RetrievalQA
from prompt_templates import similarity_chain, qna_chain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from dotenv import load_dotenv
from accelerate import disk_offload

# Imports
import chromadb
import re
import gradio as gr
import warnings
import platform

print("platform.mac_ver()  ", platform.mac_ver())


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()


new_data = load_dataset("parquet", data_files="./data/sample_300.parquet")

EMBED_MODEL_NAME = os.environ["EMBED_MODEL_NAME"]
embedding_model = SentenceTransformerEmbeddings(model_name=EMBED_MODEL_NAME)
persist_directory = "./chroma_db/similarity"


class textDoc:
    def __init__(self, text, metadata):
        self.page_content = text
        self.metadata = metadata


chroma_data = []

for data in new_data["train"]:
    chroma_data.append(textDoc(data["text_new"], {"Category": data["Category"]}))

print("Creating a ChromaDB, Ingesting the embedded train data....")


vectordb_summ = Chroma.from_documents(
    chroma_data,
    embedding=embedding_model,
    persist_directory=persist_directory,  # save the directory
)


print("Persisting the data in ChromaDB.....")

vectordb_summ.persist()  # Let's **save vectordb** so we can use it later!


MODEL_NAME = os.environ["MODEL_NAME"]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("Retrieving the Huggingface Tokenizer.....")


# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True,token=os.environ["HF_TOKEN"])
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Retrieving the pre-trained model.....", MODEL_NAME, "\n\n\n")
# device ="cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

offload_directory = "pretrained_model"

"""model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage = True,token=os.environ["HF_TOKEN"], 
            ).to(device)

"""
"""model = AutoModelForCausalLM.from_pretrained( MODEL_NAME, 
                                                trust_remote_code=True,
                                                token=os.environ["HF_TOKEN"],
                                                quantization_config=bnb_config, device_map ="auto")"""


model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="cpu")

disk_offload(model=model, offload_dir=offload_directory)

print("Here in model...")
# model.save_pretrained("models")

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.7
generation_config.top_p = 0.95
generation_config.do_sample = True
generation_config.repetition_penalty = 1.15

print("Building a Pipeline")

text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    generation_config=generation_config,
)
print("Creating the LLM Object....")
llm = HuggingFacePipeline(pipeline=text_pipeline)


# stored_vectordb = Chroma(persist_directory="chroma_db/similarity", embedding_function = embedding_model)

# vectordb_summ = stored_vectordb
similarity_search = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb_summ.as_retriever(search_kwargs={"k": 1}),
    chain_type_kwargs={"prompt": similarity_chain},
    return_source_documents=True,
)

ques_and_ans = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb_summ.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "fetch_k": 6}
    ),
    chain_type_kwargs={"prompt": qna_chain},
    return_source_documents=True,
)

question = """I have trading company and looking for technologies that can help me"""


print("Retrieving the result for a user prompt.....")
result = similarity_search.invoke({"query": question})
qna_response = result["result"].strip()
print(qna_response)
