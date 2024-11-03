from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions.actions import ActionResult
from doc_loader import load_documents
from llama_index.llm import LLMPredictor
from llama_index.core import ServiceContext
import logging

logger = logging.getLogger(__name__)

def prompt_template(question, context):
    """Constructs a prompt template for the RAG system."""
    return f"""Answer user questions based on loaded documents. 

    {context}

    1. You do not make up a story. 
    2. Keep your answer as concise as possible.
    3. Should not answer any out-of-context USER QUESTION.

    USER QUESTION: ```{question}```
    Answer in markdown:"""

async def rag(query_str, index=None, query_engine=None):
    """
    Asynchronous function to handle retrieval augmented generation (RAG) process.
    
    :param query_str: The user's query string.
    :param index: The document index, if already loaded.
    :param query_engine: The query engine for document retrieval.
    :return: ActionResult with the response or an error message.
    """
    if not query_str:
        return ActionResult(return_value="No query provided", context_updates={})
    
    if index is None or query_engine is None:
        return ActionResult(return_value="No documents loaded. Please upload documents first.", context_updates={})
    
    try:
        # Query the engine
        response = await query_engine.aquery(query_str)
        relevant_chunks = response.response

        # Construct the prompt with the template and query results
        prompt = prompt_template(query_str, relevant_chunks)
        
        # Using LLMPredictor for generation. Ensure ServiceContext and LLMPredictor are properly set up.
        service_context = ServiceContext.from_defaults(llm_predictor=LLMPredictor(llm="your_llm_model_here"))
        answer = await service_context.llm_predictor.apredict(prompt)

        logger.info(f"Generated answer for query '{query_str}': {answer}")
        
        return ActionResult(return_value=answer, context_updates={
            'last_bot_message': answer,
            '_last_bot_prompt': prompt
        })
    except Exception as e:
        error_message = f"Error in RAG process: {str(e)}"
        logger.error(error_message)
        return ActionResult(return_value="An error occurred while processing your query.", context_updates={})

def init(app: LLMRails):
    """
    Initialize the RAG pipeline with NeMo Guardrails.
    
    :param app: The LLMRails application instance.
    """
    # Note: This function might need adjustment based on how you manage the index and query_engine
    global index, query_engine
    index, query_engine = load_documents("./Config/kb")  # Ensure this returns both index and query_engine
    app.register_action(rag, "rag")
    logger.info("RAG action registered with NeMo Guardrails.")
