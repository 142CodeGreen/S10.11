import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
import shutil
import logging

logger = logging.getLogger(__name__)

def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]

def load_documents(file_objs):
    global index, query_engine  # Declare as global to modify it
    
    kb_dir = "./Config/kb"
    if not os.path.exists(kb_dir):
        os.makedirs(kb_dir)

    file_paths = get_files_from_input(file_objs) if file_objs else [kb_dir]
    documents = []
    for file_path in file_paths:
        if os.path.isfile(file_path):
            documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())
            shutil.copy2(file_path, kb_dir)
        elif os.path.isdir(file_path):
            documents.extend(SimpleDirectoryReader(input_dir=file_path).load_data())
    
    if not documents:
        logger.warning("No documents found or loaded.")
        return None

    vector_store = MilvusVectorStore(uri="milvus_demo.db", dim=1024, overwrite=True, output_fields=[])
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)
    return index, query_engine
