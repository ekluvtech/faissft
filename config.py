import os


#ollama
OLLAMA_URL= os.getenv('OLLAMA_URL','http://localhost:11434')
LLM_MODEL = os.getenv('LLM_MODEL','llama3.2')
EMBED_MODEL = os.getenv('EMBED_MODEL','mxbai-embed-large:latest')
FAISS_INDEX_NAME= os.getenv('FAISS_INDEX_NAME','faiss_idx')
# vector store config
FOLDER_PATH = os.getenv('FOLDER_PATH','/Users/srikanth/Documents/work/faissrag/data')
INDEX_STORAGE_PATH = os.getenv('INDEX_STORAGE_PATH','/Users/srikanth/Documents/work/faissrag/index')