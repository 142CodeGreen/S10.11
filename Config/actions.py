#actions.py

#from doc_index import get_index

from nemoguardrails import LLMRails, RailsConfig
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


def template(question, context, history):
    """Constructs a prompt template for the RAG system, including conversation history."""
    history_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in history])
    return f"""Answer user questions based on loaded documents and past conversation.

    Past conversation:
    {history_str}

    Current Context:
    {context}

    1. Use the information above to answer the question.
    2. You do not make up a story.
    3. Keep your answer as concise as possible.
    4. Should not answer any out-of-context USER QUESTION.

    USER QUESTION: ```{question}```
    Answer in markdown:"""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@action(is_system_action=True)
async def rag(context: Dict): #change rag to retrieve_relevant_chunks
    logger.info("rag() function called!")
    
    # Index check
    index = rails.index
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
        logger.info(f"Document context:\n{doc_context[:200]}...")  # Log first 200 characters to avoid flooding logs

        # Use the template to form the prompt including history
        prompt = template(question, doc_context, history)
        logger.info(f"Generated prompt:\n{prompt[:200]}...")  # Log first 200 characters

        # Generate the response using the LLM
        logger.info("Generating response with LLM")
        answer = await Settings.llm.complete(prompt)
        logger.info(f"LLM response: {answer.text[:200]}...")  # Log first 200 characters of the answer

        # Update context with new information
        context_updates = {
            "relevant_chunks": doc_context,
            "history": history + [(question, answer.text)]
        }

        logger.info("Returning result from rag()")
        return ActionResult(
            return_value=answer.text,
            context_updates=context_updates
        )
    except Exception as e:
        logger.error(f"Error in rag(): {str(e)}")
        return ActionResult(
            return_value="An error occurred while processing your query.",
            context_updates={}
        )

def init(app: LLMRails, index=None):
    # Store the index somewhere accessible, like setting it as an attribute of the app
    app.index = index
    app.register_action(rag, name="rag")
    #app.register_action(retrieve_relevant_chunks, name="retrieve_relevant_chunks")
    logger.info("RAG action registered successfully.")
