from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.kb.kb import KnowledgeBase
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from doc_loader import load_documents  # This should import the function from the correct module
from llama_index.core import Settings
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

async def rag(context: dict, llm: NVIDIA, kb: KnowledgeBase) -> ActionResult:
    user_message = context.get("last_user_message", "")
    context_updates = {}

    try:
        # Use KnowledgeBase to search for relevant chunks
        chunks = await kb.search_relevant_chunks(user_message)
        relevant_chunks = "\n".join([chunk["body"] for chunk in chunks])
        
        print(f"Query: {user_message}")  # Print the query
        print(f"Relevant Chunks: {relevant_chunks}")  # Print the retrieved context
        
        # Update context with the relevant chunks
        context_updates["relevant_chunks"] = relevant_chunks
        context_updates["last_bot_prompt"] = template(user_message, relevant_chunks)
        
        # Generate answer using the LLM with the constructed prompt
        answer = await llm.apredict(context_updates["last_bot_prompt"])
        context_updates["last_bot_message"] = answer
        
        return ActionResult(return_value=answer, context_updates=context_updates)
    
    except Exception as e:
        error_message = f"Unexpected error in RAG process: {str(e)}"
        logger.error(error_message)
        return ActionResult(return_value="An unexpected error occurred while processing your query.", context_updates={})

def init(app: LLMRails):
    global kb
    try:
        # Load documents and initialize KnowledgeBase
        kb = load_documents("./Config/kb")
        
        if kb is None:
            raise ValueError("Failed to initialize KnowledgeBase from loaded documents.")
        
        app.register_action(rag, name="generate_answer")
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        # Handle initialization failure as per your application's requirements
