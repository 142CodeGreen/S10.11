#actions.py

#from doc_index import get_index

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core import Settings
import asyncio
from typing import Dict
import logging

# Settings for LLM and embedding model
Settings.llm = NVIDIA(model="meta/llama-3.1-8b-instruct")
Settings.embed_model = NVIDIAEmbedding(model="NV-Embed-QA", truncate="END")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@action(is_system_action=True)
async def retrieve_relevant_chunks(context: Dict):
    logger.info("retrieve_relevant_chunks() function called!")
    
    # Index check
    index = context.get('index')
    if index is None:
        logger.error("Index not available.")
        return ActionResult(
            return_value="Index not available.",
            context_updates={}
        )
    else:
        logger.info(f"Index type: {type(index)}")

    # Retrieve question from context
    question = context.get('last_user_message', '')
    logger.info(f"User question: {question}")

    # Retrieve history from context
    history = context.get('history', [])
    logger.info(f"Conversation history: {history}")

    try:
        # Create query engine from index
        logger.info("Creating query engine")
        query_engine = index.as_query_engine()

        # Retrieve relevant contexts using the query_engine
        logger.info(f"Querying index with: {question}")
        response = await query_engine.aquery(question)

        # Log retrieved documents
        logger.info(f"Number of source nodes retrieved: {len(response.source_nodes)}")
        doc_context = "\n".join([node.text for node in response.source_nodes])
        logger.info(f"Document context:\n{doc_context[:200]}...")  # Log first 200 characters

        # Generate the response using the LLM directly without a template
        logger.info("Generating response with LLM")
        prompt = f"Answer the following question using the provided context:\n\nContext:\n{doc_context}\n\nQuestion: {question}"
        answer = await Settings.llm.complete(prompt)
        logger.info(f"LLM response: {answer.text[:200]}...")  # Log first 200 characters of the answer

        # Update context with new information
        context_updates = {
            "relevant_chunks": doc_context,
            "history": history + [(question, answer.text)]
        }

        logger.info("Returning result from retrieve_relevant_chunks()")
        return ActionResult(
            return_value=answer.text,
            context_updates=context_updates
        )
    except Exception as e:
        logger.error(f"Error in retrieve_relevant_chunks(): {str(e)}")
        return ActionResult(
            return_value="An error occurred while processing your query.",
            context_updates={}
        )

def init(app: LLMRails, index=None):
    # Store the index somewhere accessible, like setting it as an attribute of the app
    app.index = index
    app.register_action(retrieve_relevant_chunks, name="retrieve_relevant_chunks")
    logger.info("retrieve_relevant_chunks action registered successfully.")
