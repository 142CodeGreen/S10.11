# app.py

import gradio as gr
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from doc_loader import load_documents
from doc_index import get_index
from Config.actions import init
from nemoguardrails import LLMRails, RailsConfig
import logging
import asyncio
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set LLM and Embedding Model
Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")
Settings.text_splitter = SentenceSplitter(chunk_size=400, chunk_overlap=20)

kb_dir = "./Config/kb"

rails = None  # Global variable for rails, if needed

async def do_async_load_and_index(file_paths):
    try:
        # Load documents
        load_result = load_documents(*file_paths)
        
        # Indexing process
        status = await doc_index(file_paths)
        
        return load_result, status
    except Exception as e:
        logger.error(f"Exception occurred while indexing documents: {e}")
        return "Error loading documents", "Error indexing documents"

async def async_load_and_index(file_paths):
    load_result, index_status = await do_async_load_and_index(file_paths)
    return load_result, index_status

async def initialize_guardrails():
    try:
        config = RailsConfig.from_path("./Config")
        global rails
        
        # Fetch the index using get_index()
        index = get_index()
        
        if index is None:
            logger.error("Index is not available during guardrails initialization.")
            return "Guardrails not initialized: No index available.", None

        rails = LLMRails(config)
        init(rails, lambda: index)  # Pass an index provider function
        
        return "Guardrails initialized successfully.", None
    except Exception as e:
        logger.error(f"Error initializing guardrails: {e}")
        return f"Guardrails not initialized due to error: {str(e)}", None


async def stream_response(query, history):
    global rails  # Use global to access the rails variable
    if not rails:
        logger.error("Guardrails not initialized.")
        yield [("System", "Guardrails not initialized. Please load documents first.")]
        return

    try:
        user_message = {"role": "user", "content": query}
        result = await rails.generate_async(messages=[user_message])

        if isinstance(result, dict):
            if "content" in result:
                history.append((query, result["content"]))
            else:
                history.append((query, str(result)))
        else:
            if isinstance(result, str):
                history.append((query, result))
            elif hasattr(result, '__iter__'):
                for chunk in result:
                    if isinstance(chunk, dict) and "content" in chunk:
                        history.append((query, chunk["content"]))
                        yield history
                    else:
                        history.append((query, chunk))
                        yield history
            else:
                logger.error(f"Unexpected result type: {type(result)}")
                history.append((query, "Unexpected response format."))

        yield history

    except Exception as e:
        logger.error(f"Error in stream_response: {str(e)}")
        history.append(("An error occurred while processing your query.", None))
        yield history

def start_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("# RAG Chatbot for PDF Files")

        file_input = gr.File(label="Select files to upload", file_count="multiple")
        load_btn = gr.Button("Click to Load Documents")
        clear_docs_btn = gr.Button("Clear Documents")
        load_output = gr.Textbox(label="Load Status")
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Enter your question")
        clear_chat_btn = gr.Button("Clear Chat History")
        clear_all_btn = gr.Button("Clear All")

        load_btn.click(
            async_load_and_index,
            inputs=[file_input],
            outputs=[load_output, gr.Textbox(label="Index Status")]
        ).then(
            lambda: "Initializing guardrails...", None, gr.Textbox(label="Guardrail Status")
        ).then(
            initialize_guardrails,
            outputs=[gr.Textbox(label="Guardrail Status")]
        ).then(
            lambda: "Guardrails initialization complete.", None, gr.Textbox(label="Guardrail Status")
        )

        #load_btn.click(
        #    async_load_and_index,
        #    inputs=[file_input],
        #    outputs=[load_output, gr.Textbox(label="Index Status")]
        #).then(
        #    initialize_guardrails,
        #    outputs=[gr.Textbox(label="Guardrail Status")]
        #)

        
        # Function to clear documents is no longer needed; use the small "x" sign at the file input

        clear_docs_btn.click(
            lambda: ([], None, "Documents cleared"),
            inputs=[],
            outputs=[file_input, load_output]
        )

        msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot])
        clear_chat_btn.click(lambda: [], outputs=[chatbot])
        clear_all_btn.click(lambda: ([], None, "Documents and chat cleared"), inputs=[], outputs=[chatbot, file_input, load_output])

    demo.queue().launch(share=True, debug=True)

if __name__ == "__main__":
    start_gradio()
    # Or if you want to run the test when the script is run directly:
    #def test_initialize_guardrails():
        # Create a storage context with the correct persist_dir
    #    storage_context = StorageContext.from_defaults(persist_dir="./storage") 

        # Load the index using the storage context
    #    mock_index = VectorStoreIndex.from_documents([], storage_context=storage_context)

    #    result, _ = initialize_guardrails(mock_index)
    #    assert result == "Guardrails initialized successfully.", "Initialization failed"

    #test_initialize_guardrails()

    #Note: Commenting out the actual app run to prevent Gradio from launching while testing:
    #start_gradio()
