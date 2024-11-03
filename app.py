import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="llama_index")

import torch
import os
import gradio as gr
from nemoguardrails import RailsConfig, LLMRails
from llama_index.core import Settings, Document, ServiceContext
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.llms import LLMPredictor
from doc_loader import load_documents  # Assuming this returns both index and query_engine
from Config.rag_pipeline import init  # Import init for guardrails setup
import logging

Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

# Initialize LLMPredictor and ServiceContext
llm_predictor = LLMPredictor(llm=NVIDIA(model="meta/llama-3.1-8b-instruct"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=NVIDIAEmbedding(model="NV-Embed-QA", truncate="END"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


rails = None  # Global rails variable
query_engine = None  # Declare query_engine as a global variable
index = None  # Declare index as a global variable

def upload_documents(file_objs):
    """
    Upload documents and initialize the query engine.

    :param file_objs: List of file objects to upload.
    :return: Status message.
    """
    global query_engine, index
    try:
        if file_objs:
            index, query_engine = load_documents(file_objs)
            logger.info("Documents uploaded and query engine initialized.")
            return "Documents uploaded and query engine initialized successfully."
        else:
            raise ValueError("No files provided for upload.")
    except Exception as e:
        logger.error(f"Error uploading documents or initializing query engine: {str(e)}")
        return f"Error: {str(e)}"

def initiate_rails():
    """
    Initialize the Rails system.

    :return: Status message indicating if Rails were successfully initiated.
    """
    global rails, query_engine, index
    try:
        if query_engine:
            config = RailsConfig.from_path("./Config")
            rails = LLMRails(config)
            init(rails)  # Assuming init uses the global query_engine and index
            logger.info("Rails initiated successfully.")
            return "Rails initiated successfully."
        else:
            return "Query engine not found. Please upload documents first."
    except Exception as e:
        logger.error(f"Error initializing Rails: {str(e)}")
        return f"Error initializing Rails: {str(e)}"

async def stream_response(message, history):
    """
    Generate streaming responses to user queries.

    :param message: User's query string.
    :param history: Chat history to maintain context.
    :yield: Updated chat history.
    """
    if rails is None:
        yield history + [("Please initialize the system first by loading documents and initiating rails.", None)]
        return

    user_message = {"role": "user", "content": message}
    try:
        result = await rails.generate_async(messages=[user_message])
        partial_response = ""
        async for chunk in result:
            partial_response += chunk
            history.append((message, partial_response)) 
            yield history
    except Exception as e:
        logger.error(f"Error in stream_response: {str(e)}")
        yield history + [("An error occurred while processing your query.", None)]

def load_documents_and_setup(file_objs):
    """
    Load documents and set up the system.

    :param file_objs: List of file objects to upload.
    :return: A string with the status of document upload and rails initialization.
    """
    upload_status = upload_documents(file_objs)
    if "initialized successfully" in upload_status:
        rails_status = initiate_rails()
        return f"Document Upload Status: {upload_status}\nRails Initialization Status: {rails_status}"
    else:
        return f"Document Upload Status: {upload_status}"

# Set up the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# RAG Chatbot for PDF Files")
    file_input = gr.File(label="Select files to upload", file_count="multiple")
    load_btn = gr.Button("Click to Load Documents")
    load_output = gr.Textbox(label="Load Status")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Enter your question")
    clear = gr.Button("Clear")

    # Button click event to load documents and initiate rails
    load_btn.click(load_documents_and_setup, inputs=[file_input], outputs=[load_output])
    
    # Textbox submission event for querying the chatbot
    msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot])
    
    # Button click event to clear the chat
    clear.click(lambda: None, None, chatbot, queue=False)

# Launch the Gradio interface
if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)
