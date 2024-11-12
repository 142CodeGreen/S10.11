#doc_loader.py, only convert and upload documents to kb

from functools import lru_cache
import os
from llama_index.readers.file import PDFReader
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logging level

@lru_cache(maxsize=8)
def load_documents(*f_paths):

    kb_dir = "./Config/kb"
    documents = []  # Initialize an empty list to store documents
    
    # Ensure the knowledge base directory exists
    try:
        os.makedirs(kb_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Error creating directory {kb_dir}: {e}")
        return f"Failed to create knowledge base directory: {str(e)}"

    for file_path in f_paths:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            try:
                if file_path.lower().endswith(".pdf"):
                    reader = PDFReader()
                    docs = reader.load_data(file_path)
                    for i, doc in enumerate(docs):
                        markdown_filename = os.path.splitext(os.path.basename(file_path))[0] + f"_{i+1}.md"
                        markdown_filepath = os.path.join(kb_dir, markdown_filename)
                        with open(markdown_filepath, "w") as f:
                            f.write(doc.text)
                        logger.info(f"Converted and saved: {markdown_filepath}")
                        documents.append(doc)  # Add the document to the list
                else:
                    logger.info(f"Unsupported file format: {file_path}")
            except Exception as e:
                logger.error(f"Error converting document from {file_path}: {str(e)}")

    logger.info("Document conversion process completed.")
    return "Documents converted and saved successfully."
