#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
from config import *
import logging
import json
import os
import numpy as np
import faiss
import pickle

global ollm
ollm = None

global embed_model
embed_model = None

# System prompt for better context and responses
SYSTEM_PROMPT = """You are an intelligent document analysis assistant with access to various documents including technical documentation, business proposals, and security guidelines. Your role is to provide accurate, relevant, and well-structured responses based on the documents in the knowledge base.

"""

def init_llm():
    global ollm
    global embed_model
    # llm = Ollama
    # embed_model = OpenAIEmbedding(model_name="text-embedding-3-large")
    ollm = OllamaLLM(model=f"{LLM_MODEL}",base_url=f"{OLLAMA_URL}")
    embed_model = OllamaEmbeddings(base_url=f"{OLLAMA_URL}",model=f"{EMBED_MODEL}")
     

def create_embeddings(docs):
    # Create embeddings for all documents
    embeddings = []
    for doc in docs:
        embedding = embed_model.embed_query(doc.page_content)
        embeddings.append(embedding)
    return np.array(embeddings, dtype=np.float32)

def load_index():
    path = f"{FOLDER_PATH}"
    logging.info("*** Loading docs from %s", path)
    all_docs = []
    
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        logging.info("*** Loading %s", full_path)
        
        # Choose loader based on file extension
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
        
    # Create embeddings
    embeddings = create_embeddings(all_docs)
    dimension = len(embeddings[0])  # Get dimension of embeddings
    
    # Parameters for IVF-PQ
    n_data = len(embeddings)
    
    # Adjust parameters based on dataset size
    nlist = min(4, max(1, n_data // 10))  # 1 centroid per 10 vectors, at least 1, at most 4
    m = 4      # Reduced number of subquantizers for small datasets
    bits = 4   # Reduced number of bits for small datasets
    
    logging.info(f"Index parameters: nlist={nlist}, m={m}, bits={bits}, n_data={n_data}")
    
    # Initialize quantizer
    quantizer = faiss.IndexFlatL2(dimension)
    
    try:
        # Create and configure IndexIVFPQ
        index = faiss.IndexIVFPQ(quantizer,  # the coarse quantizer
                                dimension,    # dimension of the data
                                nlist,       # number of centroids for coarse quantizer
                                m,           # number of subquantizers
                                bits)        # number of bits per subquantizer
        
        # Need to train the index
        if not index.is_trained:
            logging.info("Training the index...")
            index.train(embeddings)
        
        # Add vectors to the index
        logging.info("Adding vectors to the index...")
        index.add(embeddings)
    except RuntimeError as e:
        logging.warning(f"Failed to create IndexIVFPQ: {e}")
        logging.info("Falling back to IndexFlatL2")
        # Fallback to simple IndexFlatL2 if dataset is too small
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
    
    # Save the index and documents
    os.makedirs(INDEX_STORAGE_PATH, exist_ok=True)
    index_path = os.path.join(INDEX_STORAGE_PATH, f"{FAISS_INDEX_NAME}.index")
    faiss.write_index(index, index_path)
    
    # Save document mapping
    docs_path = os.path.join(INDEX_STORAGE_PATH, f"{FAISS_INDEX_NAME}.docs")
    with open(docs_path, 'wb') as f:
        pickle.dump(all_docs, f)
    
    logging.info(f"Index saved at {index_path}")
    return index, all_docs

def search_documents(query, k=5):
    # Load the index
    index_path = os.path.join(INDEX_STORAGE_PATH, f"{FAISS_INDEX_NAME}.index")
    docs_path = os.path.join(INDEX_STORAGE_PATH, f"{FAISS_INDEX_NAME}.docs")
    
    index = faiss.read_index(index_path)
    with open(docs_path, 'rb') as f:
        docs = pickle.load(f)
    
    # Create query embedding with system prompt context
    augmented_query = f"{SYSTEM_PROMPT}\n\nUser Query: {query}"
    query_vector = np.array([embed_model.embed_query(augmented_query)], dtype=np.float32)
    
    # Set search parameters based on index type
    if isinstance(index, faiss.IndexIVFPQ):
        index.nprobe = 2  # Number of cells to visit during search
        distance_threshold = 0.8
    else:
        # For IndexFlatL2, use a different threshold since distances are computed differently
        distance_threshold = 1.5
    
    # Search
    distances, indices = index.search(query_vector, k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:  # Valid index
            # Only include results with distance below threshold (smaller distance = more relevant)
            if float(distances[0][i]) < distance_threshold:  # Convert to Python float
                results.append({
                    'content': docs[idx].page_content,
                    'metadata': docs[idx].metadata,
                    'distance': float(distances[0][i]),  # Convert to Python float
                    'confidence_score': 1 - float(distances[0][i])  # Convert distance to confidence
                })
    
    return results

def query_pdf(query):
    # Load document using PyPDFLoader document loader
   
    # Load from local storage
    persisted_vectorstore = FAISS.load_local(f"{FAISS_INDEX_NAME}", embed_model,allow_dangerous_deserialization=True)
    
    qa = RetrievalQA.from_chain_type(llm=ollm, chain_type="stuff", retriever=persisted_vectorstore.as_retriever())
    result = qa.invoke(query)
    json_str = json.dumps(result, indent=4)
    print(json_str)

def format_search_results(results):
    """Format search results into a structured summary"""
    if not results:
        return {
            "status": "no_results",
            "message": "No relevant documents found for your query.",
            "results": []
        }
    
    formatted_results = {
        "status": "success",
        "summary": {
            "total_results": len(results),
            "average_confidence": sum(r['confidence_score'] for r in results) / len(results),
            "sources": list(set(r['metadata'].get('source', 'Unknown') for r in results))
        },
        "results": []
    }
    
    for idx, result in enumerate(results, 1):
        formatted_result = {
            "result_id": idx,
            "content": result['content'],
            "metadata": {
                "source": result['metadata'].get('source', 'Unknown'),
                "page": result['metadata'].get('page', 'N/A'),
                "confidence_score": round(result['confidence_score'], 4)
            }
        }
        formatted_results["results"].append(formatted_result)
    
    return formatted_results

def main():
    init_llm()
    folder_path = f"{FOLDER_PATH}"
    index, docs = load_index()
    logging.info(f"Created index with {len(docs)} documents")
    
    print("\nDocument Search System")
    print("=====================")
    print(f"Indexed {len(docs)} documents")
    print("\nEnter your query (type 'exit' to quit):")
    
    query = input("\nQuery: ")
    while query.lower() != "exit":
        results = search_documents(query)
        formatted_output = format_search_results(results)
        
        # Print formatted results
        print("\nSearch Results")
        print("==============")
        
        if formatted_output["status"] == "no_results":
            print(formatted_output["message"])
        else:
            # Print summary
            print("\nSummary:")
            print(f"Total Results: {formatted_output['summary']['total_results']}")
            print(f"Average Confidence: {formatted_output['summary']['average_confidence']:.4f}")
            print(f"Sources: {', '.join(formatted_output['summary']['sources'])}")
            
            # Print individual results
            print("\nDetailed Results:")
            for result in formatted_output["results"]:
                print(f"\n[Result {result['result_id']}]")
                print(f"Source: {result['metadata']['source']}")
                print(f"Page: {result['metadata']['page']}")
                print(f"Confidence: {result['metadata']['confidence_score']}")
                print("-" * 80)
                print(result['content'])
                print("-" * 80)
        
        query = input("\nQuery (type 'exit' to quit): ")

if __name__ == "__main__":
    main()
