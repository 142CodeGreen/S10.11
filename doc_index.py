# indexer.py

from doc_loader import load_documents
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
import logging
import asyncio
import os

# Configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

kb_dir = "./Config/kb"

# Set up the text splitter and embedding model
Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.text_splitter = SentenceSplitter(chunk_size=400)
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

global_index = None

async def doc_index(file_paths, storage_context=None):
    global index
    try:
        logger.debug("Starting document indexing process.")

        # Check if kb_dir has any .md files
        if not any(file.endswith('.md') for file in os.listdir(kb_dir)):
            logger.info("No .md files found in the knowledge base directory. Uploading documents.")
            # Assuming upload_documents returns a status message
            load_status = await load_documents(*file_paths)
            logger.info(f"Document upload status: {load_status}")
            # After uploading, we should re-check if there are documents now
            documents = SimpleDirectoryReader(kb_dir, required_exts=['.md']).load_data()
        else:
            logger.info(".md files found, proceeding to indexing.")
            documents = SimpleDirectoryReader(kb_dir, required_exts=['.md']).load_data()

        if not documents:
            logger.info("No documents were processed for indexing after upload attempt.")
            return None, "No documents available to index."

        logger.debug(f"Number of documents loaded: {len(documents)}")

        # Use provided storage context or create a new in-memory one
        if storage_context is None:
            vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
        else:
            # Do not attempt to set vector_store here again, it should be already set
            pass
            
        # Indexing logic
        #vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True)
        #storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index from the documents in the kb_dir
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        logger.info("Documents indexed successfully.")
        logger.info(f"Index object: {index}")  # Add this line to inspect the index object


        # Sample query after indexing for verification
        query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)
        sample_query = "What is the document about?"
        sample_response = await query_engine.aquery(sample_query)
        logger.info(f"Sample query result: {sample_query}\n{sample_response.get_formatted_sources()}")

        # Save the index 
        storage_context.persist(persist_dir="./storage")  # Choose a directory to save to
        logger.info("Storage context saved to disk.")

        # Return the index
        logger.info(f"Index created: {index}")
        global_index = index
        return index, "Documents indexed successfully"
        #return index         # "Documents indexed successfully"

    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        return None, f"Failed to index documents: {str(e)}"

def get_index():
    global global_index
    if global_index is None:
        logger.info("Index has not been created yet.")
    return global_index
