import os
import shutil
import logging
from functools import lru_cache
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from nemoguardrails.kb.kb import KnowledgeBase
from pdf2image import convert_from_path  # For PDF to image conversion (for OCR)
#import pytesseract  # For OCR
#from PIL import Image  # For image processing
#import mammoth  # For docx to markdown
#from bs4 import BeautifulSoup  # For HTML to markdown
#import markdownify  # For converting HTML to Markdown
#import pandas as pd  # For csv conversion

# Configure logging
logger = logging.getLogger(__name__)
Settings.text_splitter = SentenceSplitter(chunk_size=400, chunk_overlap=20)

# Function to convert different file types to Markdown
def convert_file_to_markdown(file_path):
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == '.pdf':
        # Convert PDF to images, then use OCR
        images = convert_from_path(file_path)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image)
        markdown_path = os.path.join("./Config/kb", os.path.splitext(os.path.basename(file_path))[0] + ".md")
        with open(markdown_path, 'w', encoding='utf-8') as outfile:
            outfile.write(text)
        return markdown_path

    #elif file_extension in ['.doc', '.docx']:
        # Convert docx to markdown
    #    with open(file_path, "rb") as docx_file:
    #        result = mammoth.convert_to_html(docx_file)
    #        html_content = result.value
    #    markdown_content = markdownify.markdownify(html_content)
    #    markdown_path = os.path.join("./Config/kb", os.path.splitext(os.path.basename(file_path))[0] + ".md")
    #    with open(markdown_path, 'w', encoding='utf-8') as outfile:
    #        outfile.write(markdown_content)
    #    return markdown_path

    #elif file_extension == '.csv':
        # Convert CSV to Markdown table
    #    df = pd.read_csv(file_path)
    #    markdown = df.to_markdown()
    #    markdown_path = os.path.join("./Config/kb", os.path.splitext(os.path.basename(file_path))[0] + ".md")
    #    with open(markdown_path, 'w', encoding='utf-8') as outfile:
    #        outfile.write(markdown)
    #    return markdown_path

    #elif file_extension in ['.jpg', '.png']:
        # For images, we might just want to describe them or use OCR if needed
    #    image = Image.open(file_path)
    #    text = pytesseract.image_to_string(image)
    #    markdown_path = os.path.join("./Config/kb", os.path.splitext(os.path.basename(file_path))[0] + ".md")
    #    with open(markdown_path, 'w', encoding='utf-8') as outfile:
    #        outfile.write(f"![]({file_path})\n\nThis image contains the following text:\n{text}")
    #    return markdown_path

    #elif file_extension == '.html':
        # Convert HTML to Markdown
    #    with open(file_path, 'r', encoding='utf-8') as file:
    #        html_content = file.read()
    #    markdown_content = markdownify.markdownify(html_content)
    #    markdown_path = os.path.join("./Config/kb", os.path.splitext(os.path.basename(file_path))[0] + ".md")
    #    with open(markdown_path, 'w', encoding='utf-8') as outfile:
    #        outfile.write(markdown_content)
    #    return markdown_path

    else:
        logger.warning(f"File extension {file_extension} not supported for conversion.")
        return None

@lru_cache(maxsize=1)
def cached_load_documents(file_paths):
    file_paths = tuple(file_paths) if isinstance(file_paths, (list, tuple)) else (file_paths,)
    documents = []
    kb_dir = "./Config/kb"
    if not os.path.exists(kb_dir):
        os.makedirs(kb_dir)

    for file_path in file_paths:
        try:
            markdown_path = convert_file_to_markdown(file_path)
            if markdown_path:
                documents.extend(SimpleDirectoryReader(input_files=[markdown_path]).load_data())
        except Exception as e:
            logger.error(f"Error loading or converting document from {file_path}: {str(e)}")

    if not documents:
        logger.warning("No documents found or loaded.")
    return documents

def load_documents(file_paths):
    kb = None  # Initialize kb

    # If file_paths is a string, convert it to a list
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    for file_path in file_paths:
        documents = cached_load_documents(file_path)  # Call with a single file path

        if not documents:
            continue  # Skip to the next file if no documents were loaded

        try:
            vector_store = MilvusVectorStore(uri="milvus_demo.db", dim=1024, overwrite=True, output_fields=[])
            
            # Create a new KnowledgeBase instance for each document
            kb = KnowledgeBase(vector_store=vector_store)  
            
            for doc in documents:
                kb.add_document(doc)
            
            logger.info(f"Document {file_path} loaded into KnowledgeBase.")
        except Exception as e:
            logger.error(f"Error during KnowledgeBase setup: {str(e)}")
            return None  # Or handle the error as needed

    return kb  # Return the KnowledgeBase
