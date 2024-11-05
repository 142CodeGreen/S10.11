from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions.actions import ActionResult
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from doc_loader import load_documents  # This should import the function from the correct module
from llama_index.core import Settings
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set LLM and Embedding Model
Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

def template(question, context):
    """Constructs a prompt template for the RAG system."""
    return f"""Answer user questions based on loaded documents. 

    {context}

    1. You do not make up a story. 
    2. Keep your answer as concise as possible.
    3. Should not answer any out-of-context USER QUESTION.

    USER QUESTION: ```{question}```
    Answer in markdown:"""


async def rag(context: dict, llm: NVIDIA, query_engine):
    context_updates = {}
    message = context.get('last_user_message', '')

    try:
        response = await query_engine.aquery(message)
        relevant_chunks = response.response
        context_updates["relevant_chunks"] = relevant_chunks
        context_updates["_last_bot_prompt"] = template(message, relevant_chunks)
        answer = await llm.apredict(context_updates["_last_bot_prompt"])
        context_updates["last_bot_message"] = answer
        return ActionResult(return_value=answer, context_updates=context_updates)
    
    except Exception as e:
        error_message = f"Unexpected error in RAG process: {str(e)}"
        logger.error(error_message)
        return ActionResult(return_value="An unexpected error occurred while processing your query.", context_updates={})

def init(app: LLMRails):
    global index, query_engine
    index, query_engine = load_documents("./Config/kb")
    if index is None or query_engine is None:
        logger.error("Failed to load documents or create query engine.")
        return

    app.register_action(rag, name="self_check_hallucination")  # Register the action 
