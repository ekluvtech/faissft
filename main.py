import faiss
import os
import uuid
import logging
import json
import numpy as np
import pickle
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
from langchain_community.docstore.in_memory import InMemoryDocstore
from config import *

# Initialize global variables
global ollm
ollm = None
global embed_model
embed_model = None

# System prompt for better context and responses
SYSTEM_PROMPT = """You are an intelligent document analysis assistant with access to various documents including technical documentation, business proposals, and security guidelines. Your role is to provide accurate, relevant, and well-structured responses based on the documents in the knowledge base."""

def init_llm():
    global ollm, embed_model
    ollm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_URL)
    embed_model = OllamaEmbeddings(base_url=OLLAMA_URL, model=EMBED_MODEL)

def create_embeddings(docs):
    embeddings = [embed_model.embed_query(doc.page_content) for doc in docs]
    return np.array(embeddings, dtype=np.float32)

def load_index():
    path = FOLDER_PATH
    logging.info("*** Loading docs from %s", path)
    all_docs = []
    
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        logging.info("*** Loading %s", full_path)
        
        file_extension = os.path.splitext(entry)[1].lower()
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(full_path)
            elif file_extension == '.txt':
                loader = TextLoader(full_path, encoding='utf-8')
            else:
                logging.warning(f"Unsupported file type: {file_extension}. Skipping {entry}")
                continue
                
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
            docs = text_splitter.split_documents(documents=documents)
            all_docs.extend(docs)
        except Exception as e:
            logging.error(f"Error loading {entry}: {str(e)}")
            continue
    
    if not all_docs:
        raise ValueError("No documents were successfully loaded. Please check your input files.")
        
    embeddings = create_embeddings(all_docs)
    dimension = len(embeddings[0])
    
    n_data = len(embeddings)
    nlist = min(4, max(1, n_data // 10))
    m = 4
    bits = 4
    
    logging.info(f"Index parameters: nlist={nlist}, m={m}, bits={bits}, n_data={n_data}")
    
    quantizer = faiss.IndexFlatL2(dimension)
    try:
        index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, bits)
        if not index.is_trained:
            logging.info("Training the index...")
            index.train(embeddings)
        logging.info("Adding vectors to the index...")
        index.add(embeddings)
    except RuntimeError as e:
        logging.warning(f"Failed to create IndexIVFPQ: {e}")
        logging.info("Falling back to IndexFlatL2")
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
    
    os.makedirs(INDEX_STORAGE_PATH, exist_ok=True)
    index_path = os.path.join(INDEX_STORAGE_PATH, f"{FAISS_INDEX_NAME}.index")
    faiss.write_index(index, index_path)
    
    docs_path = os.path.join(INDEX_STORAGE_PATH, f"{FAISS_INDEX_NAME}.docs")
    with open(docs_path, 'wb') as f:
        pickle.dump(all_docs, f)
    
    # Create and save FAISS vector store
   
    vectorstore = FAISS(
        embedding_function=embed_model,
        index=index,
        docstore=InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)}),
        index_to_docstore_id={i: str(i) for i in range(len(docs))}
    )
    vectorstore.save_local(INDEX_STORAGE_PATH, FAISS_INDEX_NAME)
    logging.info(f"Index saved at {index_path}, vector store saved at {INDEX_STORAGE_PATH}/{FAISS_INDEX_NAME}.faiss")
    return index, all_docs

def query_pdf(query):
    try:
        persisted_vectorstore = FAISS.load_local(
            INDEX_STORAGE_PATH,
            embed_model,
            FAISS_INDEX_NAME,
            allow_dangerous_deserialization=True
        )
        qa = RetrievalQA.from_chain_type(
            llm=ollm,
            chain_type="stuff",
            retriever=persisted_vectorstore.as_retriever()
        )
        result = qa.invoke(query)
        print(result)
        return {
            "answer": result["result"],
            #"sources": [doc.metadata.get('source', 'Unknown') for doc in result["source_documents"]]
        }
    except Exception as e:
        logging.error(f"Error in query_pdf: {str(e)}")
        return {"answer": "Error processing query.", "sources": []}

def main():
    init_llm()
    # Uncomment to recreate index if needed
    load_index()
    # logging.info(f"Created index with {len(docs)} documents")
    
    print("\nDocument Search System")
    print("=====================")
    print("\nEnter your query (type 'exit' to quit):")
    
    query = input("\nQuery: ")
    while query.lower() != "exit":
        if not query:
            print("Please enter a valid query.")
            query = input("\nQuery: ")
            continue
        qa_result = query_pdf(query)
        print("\nQA Response")
        print("===========")
        print(f"Answer: {qa_result['answer']}")
        query = input("\nQuery (type 'exit' to quit): ")

if __name__ == "__main__":
    main()
