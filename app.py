import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="llama_index")

import gradio as gr
from nemoguardrails import RailsConfig, LLMRails
from llama_index.core import Settings
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from doc_loader import load_documents
from Config.actions import init
import logging
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set LLM and Embedding Model
Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

rails = None  # Global rails variable
kb = None  # Global KnowledgeBase variable

def upload_documents(file_objs):
    global kb
    try:
        if file_objs:
            kb = None  # Initialize kb
            for file_obj in file_objs:
                kb = load_documents(file_obj.name)  # Load each file individually
            if kb is None:
                raise ValueError("Failed to load any documents.")
            return "Documents uploaded and KnowledgeBase initialized successfully."
        else:
            raise ValueError("No files provided for upload.")
    except Exception as e:
        logger.error(f"Error uploading documents or initializing KnowledgeBase: {str(e)}")
        return f"Error: {str(e)}"

async def load_documents_and_setup(file_objs):
    global rails, kb
    try:
        upload_status = upload_documents(file_objs)
        if "initialized successfully" in upload_status:
            config = RailsConfig.from_path("./Config")
            rails = LLMRails(config)
            
            if kb is None:
                logger.error("Failed to initialize KnowledgeBase.")
                return f"Document Upload Status: {upload_status}\nRails Initialization Status: KnowledgeBase initialization failed."
            
            init(rails, kb)  # Initialize rails with the KnowledgeBase
            return f"Document Upload Status: {upload_status}\nRails Initialization Status: Rails initiated successfully."
        else:
            return f"Document Upload Status: {upload_status}"
    except Exception as e:
        logger.error(f"Error in document loading and setup: {str(e)}")
        return f"Error in document loading and setup: {str(e)}"

async def stream_response(message, history):
    if rails is None or kb is None:
        yield history + [("Please initialize the system first by loading documents and initiating rails.", None)]
        return
    
    user_message = {"role": "user", "content": message}
    try:
        result = await rails.generate_async(messages=[user_message], kb=kb)
        if isinstance(result, dict):
            if "content" in result:
                yield history + [(message, result["content"])]
            else:
                yield history + [(message, str(result))]
        else:
            if isinstance(result, str):
                yield history + [(message, result)]
            else:
                for chunk in result:
                    if isinstance(chunk, dict) and "content" in chunk:
                        yield history + [(message, chunk["content"])]
                    else:
                        yield history + [(message, chunk)]
    except Exception as e:
        logger.error(f"Error in stream_response: {str(e)}")
        yield history + [("An error occurred while processing your query.", None)]

def start_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("# RAG Chatbot for PDF Files")
        file_input = gr.File(label="Select files to upload", file_count="multiple")
        load_btn = gr.Button("Click to Load Documents")
        load_output = gr.Textbox(label="Load Status")
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Enter your question")
        clear = gr.Button("Clear")

        load_btn.click(load_documents_and_setup, inputs=[file_input], outputs=[load_output])
        msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue().launch(share=True, debug=True)

if __name__ == "__main__":
    start_gradio()
