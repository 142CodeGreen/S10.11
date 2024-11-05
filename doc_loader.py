import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from nemoguardrails.kb.kb import KnowledgeBase
import shutil
import logging
from functools import lru_cache  # For in-memory caching

logger = logging.getLogger(__name__)
Settings.text_splitter = SentenceSplitter(chunk_size=400, chunk_overlap=20)


@lru_cache(maxsize=1)
def cached_load_documents(file_paths):
    file_paths = tuple(file_paths) if isinstance(file_paths, (list, tuple)) else (file_paths,)
    documents = []
    kb_dir = "./Config/kb"
    if not os.path.exists(kb_dir):
        os.makedirs(kb_dir)

    for file_path in file_paths:
        try:
            if os.path.isfile(file_path):
                documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())
                shutil.copy2(file_path, kb_dir)
            elif os.path.isdir(file_path):
                documents.extend(SimpleDirectoryReader(input_dir=file_path).load_data())
        except Exception as e:
            logger.error(f"Error loading document from {file_path}: {str(e)}")

    if not documents:
        logger.warning("No documents found or loaded.")
    return documents

def load_documents(file_paths):
    """
    Load documents into a KnowledgeBase using Milvus for vector storage.

    Args:
    file_paths (str or list): Path(s) to document file(s) or directory(ies).

    Returns:
    KnowledgeBase: An instance of KnowledgeBase with loaded documents.
    """
    documents = cached_load_documents(file_paths)

    if not documents:
        return None

    try:
        # Initialize MilvusVectorStore
        vector_store = MilvusVectorStore(uri="milvus_demo.db", dim=1024, overwrite=True, output_fields=[])
        
        # Create or initialize KnowledgeBase with Milvus
        kb = KnowledgeBase(vector_store=vector_store)
        
        # Add documents to KnowledgeBase
        for doc in documents:
            # Assuming KnowledgeBase has an add_document method that interacts with Milvus
            kb.add_document(doc)
        
        logger.info("Documents loaded into KnowledgeBase with Milvus vector store.")
        return kb
    except Exception as e:
        logger.error(f"Error during KnowledgeBase creation with Milvus: {str(e)}")
        return None
