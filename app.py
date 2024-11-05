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
query_engine = None  # Global query_engine variable
index = None  # Global index variable

def upload_documents(file_objs):
    global query_engine, index
    try:
        if file_objs:
            for file_obj in file_objs:  # Iterate over the files
                index, query_engine = load_documents(file_obj)  # Load each file individually
            #index, query_engine = load_documents(file_objs)
            return "Documents uploaded and query engine initialized successfully."
        else:
            raise ValueError("No files provided for upload.")
    except Exception as e:
        logger.error(f"Error uploading documents or initializing query engine: {str(e)}")
        return f"Error: {str(e)}"


@lru_cache(maxsize=1)
def load_documents(file_path):  # Change argument to file_path (singular)
    global index, query_engine
    
    kb_dir = "./Config/kb"
    if not os.path.exists(kb_dir):
        os.makedirs(kb_dir)

    documents = []
    if os.path.isfile(file_path):
        documents.extend(SimpleDirectoryReader(input_files=[file_path]).load_data())
        shutil.copy2(file_path, kb_dir)  # Copy the single file
    elif os.path.isdir(file_path):
        documents.extend(SimpleDirectoryReader(input_dir=file_path).load_data())
       
async def load_documents_and_setup(file_objs):
    global rails, query_engine, index
    try:
        upload_status = upload_documents(file_objs)
        if "initialized successfully" in upload_status:
            config = RailsConfig.from_path("./Config")
            rails = LLMRails(config)

            # --- Load the index from cache or create it ---
            index, query_engine = load_documents(file_objs)
            init(rails)  # Initialize rails with the new document context
            return f"Document Upload Status: {upload_status}\nRails Initialization Status: Rails initiated successfully."
        else:
            return f"Document Upload Status: {upload_status}"
    except Exception as e:
        logger.error(f"Error in document loading and setup: {str(e)}")
        return f"Error in document loading and setup: {str(e)}"

async def stream_response(message, history):
    if rails is None:
        yield history + [("Please initialize the system first by loading documents and initiating rails.", None)]
        return

    if query_engine is None or index is None:
        yield history + [("Please upload documents first.", None)]
        return
    
    user_message = {"role": "user", "content": message}
    try:
        # Initiate the NeMo Guardrails flow
        result = await rails.generate_async(messages=[user_message])
        
        # If result is a dictionary or contains 'content', handle appropriately
        if isinstance(result, dict):
            if "content" in result:
                yield history + [(message, result["content"])]
            else:
                # If no 'content' key, maybe yield the whole dict or handle errors
                yield history + [(message, str(result))]
        else:
            # Assuming result could be a string or an iterable of chunks
            if isinstance(result, str):
                yield history + [(message, result)]
            else:
                # For an iterable result
                for chunk in result:
                    if isinstance(chunk, dict) and "content" in chunk:
                        yield history + [(message, chunk["content"])]
                    else:
                        yield history + [(message, chunk)]
    except Exception as e:
        logger.error(f"Error in stream_response: {str(e)}")
        yield history + [("An error occurred while processing your query.", None)]
        

#async def stream_response(message, history):
#    if rails is None:
#        yield history + [("Please initialize the system first by loading documents and initiating rails.", None)]
#        return

#    if query_engine is None or index is None:
#        yield history + [("Please upload documents first.", None)]
#        return
    
#    user_message = {"role": "user", "content": message}
#    try:
#        result = await rails.generate_async(messages=[user_message])
#        partial_response = ""
#        async for chunk in result:
#            partial_response += chunk
#            history.append((message, partial_response)) 
#            yield history
#    except Exception as e:
#        logger.error(f"Error in stream_response: {str(e)}")
#        yield history + [("An error occurred while processing your query.", None)]


def start_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("# RAG Chatbot for PDF Files")
        file_input = gr.File(label="Select files to upload", file_count="multiple")
        load_btn = gr.Button("Click to Load Documents")
        load_output = gr.Textbox(label="Load Status")
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Enter your question")
        clear = gr.Button("Clear")

        # Load documents and setup rails
        load_btn.click(load_documents_and_setup, inputs=[file_input], outputs=[load_output])
        
        # Handle user query
        msg.submit(stream_response, inputs=[msg, chatbot], outputs=[chatbot])
        
        # Clear chat history
        clear.click(lambda: None, None, chatbot, queue=False)

    # Launch the Gradio interface
    demo.queue().launch(share=True, debug=True)

if __name__ == "__main__":
    start_gradio()
