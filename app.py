import torch
import os
import gradio as gr
import logging
from nemoguardrails import RailsConfig
from nemoguardrails.llm.providers import get_llm_provider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter

Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
Settings.text_splitter = SentenceSplitter(chunk_size=400)

# Ensure GPU usage
if torch.cuda.is_available():
    logger.info("GPU is available and will be used.")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  
else:
    logger.warning("GPU not detected or not configured correctly. Falling back to CPU.")

index = None
query_engine = None
rails = None

# Function to get file names from file objects
def get_files_from_input(file_objs):
    if not file_objs:
        return []
    return [file_obj.name for file_obj in file_objs]

# Function to load documents, create the index, and initialize guardrails
def load_documents(file_objs):
    global index, query_engine, rails
    try:
        if not file_objs:
            return "Error: No files selected."

        file_paths = get_files_from_input(file_objs)
        documents = []
        for file_path in file_paths:
            directory = os.path.dirname(file_path)
            documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())

        if not documents:
            return f"No documents found in the selected files."

        # Create a Milvus vector store and storage context
        vector_store = MilvusVectorStore(
            host="127.0.0.1",
            port=19530,
            dim=1024,
            collection_name="your_collection_name",
            gpu_id=0  # Specify the GPU ID to use
        )
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create the index from the documents
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        # Create the query engine after the index is created
        query_engine = index.as_query_engine(similarity_top_k=20, streaming=True)

        # Initialize NeMo Guardrails after documents are loaded
        config = RailsConfig.from_path("./nemo")
        llm_provider = get_llm_provider(model_name="meta/llama-3.1-8b-instruct")
        rails = RailsContext(config=config, llm_provider=llm_provider)

        # Register RAG execution with NeMo Guardrails
        rails.register_action("rag", lambda context, statements: query_engine.query(context.get("user_input")))

        return f"Successfully loaded {len(documents)} documents from {len(file_paths)} files."
    except Exception as e:
        return f"Error loading documents: {str(e)}"

# Function to handle chat interactions using NeMo Guardrails
def chat(message, history):
    global rails
    if rails is None:
        return history + [("Please upload a file first.", None)]
    try:
        # Execute the entire flow including checks
        rails.execute(message)
        return history  # Assuming RailsContext modifies history
    except Exception as e:
        return history + [(message, f"Error processing query: {str(e)}")]

# Function to stream responses
def stream_response(message, history):
    global rails
    if rails is None:
        yield history + [("Please upload a file first.", None)]
        return
    try:
        yield from rails.stream(message)  # Stream the response
    except Exception as e:
        yield history + [(message, f"Error processing query: {str(e)}")]

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# RAG Chatbot with Guardrails")

    with gr.Row():
        file_input = gr.File(label="Select files to upload", file_count="multiple")
        load_btn = gr.Button("Load Documents")

    load_output = gr.Textbox(label="Load Status")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Enter your question", interactive=True)
    clear = gr.Button("Clear")

    load_btn.click(load_documents, inputs=[file_input], outputs=[load_output])
    msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)
