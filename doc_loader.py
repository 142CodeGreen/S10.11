import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
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
    Load documents, create or retrieve an index, and setup a query engine.

    Args:
    file_paths (str or list): Path(s) to document file(s) or directory(ies).

    Returns:
    tuple: A tuple containing the VectorStoreIndex and the QueryEngine.
    """
    documents = cached_load_documents(file_paths)
    
    if not documents:
        return None, None

    try:
        vector_store = MilvusVectorStore(uri="milvus_demo.db", dim=1024, overwrite=True, output_fields=[])
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)

        logger.info("Index and Query Engine created or retrieved from cache.")
        
        # Test Query - Commented out for production use
        # test_query = "This is a test query"
        # test_response = query_engine.query(test_query)
        # print(f"Test Query Context: {test_response}")

        return index, query_engine
    except Exception as e:
        logger.error(f"Error during index creation: {str(e)}")
        return None, None
