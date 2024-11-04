from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions.actions import ActionResult
from doc_loader import load_documents
from llama_index.core import Settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def template(question, context):
    """Constructs a prompt template for the RAG system."""
    return f"""Answer user questions based on loaded documents. 

    {context}

    1. You do not make up a story. 
    2. Keep your answer as concise as possible.
    3. Should not answer any out-of-context USER QUESTION.

    USER QUESTION: ```{question}```
    Answer in markdown:"""

async def rag(index=None, query_engine=None, context=None):
    """
    Asynchronous function to handle retrieval augmented generation (RAG) process.
    
    :param index: The document index, if already loaded.
    :param query_engine: The query engine for document retrieval.
    :param context: Context containing user message.
    :return: ActionResult with the response or an error message.
    """
    if context is None:
        return ActionResult(return_value="Context not provided", context_updates={})
    
    message = context.get('last_user_message', '')
    if not message:
        return ActionResult(return_value="No query provided", context_updates={})
    
    if index is None or query_engine is None:
        return ActionResult(return_value="No documents loaded. Please upload documents first.", context_updates={})
    
    try:
        response = await query_engine.aquery(message)
        relevant_chunks = response.response
        prompt = template(message, relevant_chunks)
        answer = await Settings.llm.apredict(prompt)
        
        logger.info(f"Generated answer for query '{message}': {answer}")
        return ActionResult(return_value=answer, context_updates={
            'last_bot_message': answer,
            '_last_bot_prompt': prompt
        })
    except AttributeError as e:
        error_message = f"Error accessing query engine or LLM: {str(e)}"
        logger.error(error_message)
        return ActionResult(return_value="An error occurred while accessing the system.", context_updates={})
    except Exception as e:
        error_message = f"Unexpected error in RAG process: {str(e)}"
        logger.error(error_message)
        return ActionResult(return_value="An unexpected error occurred while processing your query.", context_updates={})

def init(app: LLMRails):
    """
    Initialize the RAG pipeline with NeMo Guardrails.
    
    :param app: The LLMRails application instance.
    """
    global index, query_engine
    index, query_engine = load_documents("./Config/kb")
    app.register_action(rag, "rag")
    logger.info("RAG action registered with NeMo Guardrails.")
