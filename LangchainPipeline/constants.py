import os 
import chromadb
from chromadb.config import Settings 
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory='db',
        # anonymized_telemetry=False
)

# from pydantic_settings import BaseSettings

# class ChromaSettings(BaseSettings):
#     chroma_db_impl: str
#     persist_directory: str
#     anonymized_telemetry: bool

#     class Config:
#         extra = "forbid"  # Ensures no extra fields are allowed

# # Use the ChromaSettings class to create an instance of settings
# CHROMA_SETTINGS = ChromaSettings(
#     chroma_db_impl='duckdb+parquet',
#     persist_directory='db',
#     anonymized_telemetry=False
# )

