from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions.actions import ActionResult
from doc_loader import load_documents # loaded_query_engine
import logging

logger = logging.getLogger(__name__)

def prompt_template(question, context):
    return f"""Answer user questions based on loaded documents. 

    {context}

    1. You do not make up a story. 
    2. Keep your answer as concise as possible.
    3. Should not answer any out-of-context USER QUESTION.

    USER QUESTION: ```{question}```
    Answer in markdown:"""


async def rag(context: dict, llm):
    message = context.get('last_user_message', '')
    if not message:
        return ActionResult(return_value="No query provided", context_updates={})
    
    try:
        global query_engine
        if not query_engine:
            logger.warning("Query engine not initialized. Attempting initialization.")
            _, query_engine = load_documents("./Config/kb")  # Load documents and initialize query_engine
            logger.info("Query engine initialized.")

        if query_engine is None:
            logger.error("Failed to initialize query engine.")
            return ActionResult(return_value="Failed to initialize query engine.", context_updates={})

        # Query the engine
        response = await query_engine.aquery(message)
        relevant_chunks = response.response

        # Construct the prompt with the template and query results
        prompt = prompt_template(message, relevant_chunks)
        answer = await llm.apredict(prompt)
        
        logger.info(f"Generated answer for query '{message}': {answer.text}")
        
        return ActionResult(return_value=answer.text, context_updates={
            'last_bot_message': answer.text,
            '_last_bot_prompt': prompt
        })
    except Exception as e:
        error_message = f"Error in RAG process: {str(e)}"
        logger.error(error_message)
        return ActionResult(return_value="An error occurred while processing your query.", context_updates={})

def init(app: LLMRails):
    global query_engine
    app.register_action(rag, "rag")
    logger.info("RAG action registered with NeMo Guardrails.")
