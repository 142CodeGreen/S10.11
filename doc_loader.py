import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
import shutil
import logging

logger = logging.getLogger(__name__)
Settings.text_splitter = SentenceSplitter(chunk_size=400, chunk_overlap=20)

async def load_documents(file_paths):
    global index, query_engine  # Declare as global to modify it
    
    kb_dir = "./Config/kb"
    if not os.path.exists(kb_dir):
        os.makedirs(kb_dir)

    file_paths = file_paths if isinstance(file_paths, list) else [file_paths]
    
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

    vector_store = await MilvusVectorStore(uri="milvus_demo.db", dim=1024, overwrite=True, output_fields=[])
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    query_engine = await index.as_query_engine(similarity_top_k=20, streaming=True)
    # --- Test Query ---
    test_query = "This is a test query"  # Replace with a relevant query
    test_response = query_engine.query(test_query)
    print(f"Test Query Context: {test_response}")
    
    return index, query_engine
