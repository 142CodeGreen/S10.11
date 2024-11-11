#actions.py

from doc_index import get_index
#from doc_index import doc_index

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions.actions import ActionResult
from nemoguardrails.actions.actions import action
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core import Settings
import asyncio
from typing import Dict

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

@action(is_system_action=True)
async def rag(context: Dict):
    print("rag() function called!")
    index = get_index()  # Get the pre-existing index
    #index = context.get('index')  # Get the index from the context
    
    if index is None:
        return ActionResult(
            return_value=f"Index not available. {status}",
            context_updates={}
        )

    question = context.get('last_user_message', '')
    history = context.get('history', [])

    try:
        # Create query engine from global_index
        query_engine = index.as_query_engine()

        # Retrieve relevant contexts using the query_engine
        response = await query_engine.aquery(question)

        # Create context from retrieved documents
        doc_context = "\n".join([node.text for node in response.source_nodes])

        # Use the template to form the prompt including history
        prompt = template(question, doc_context, history)

        # Generate the response using the LLM (Assuming LLM is configured in Settings)
        answer = await Settings.llm.complete(prompt)

        # Update context with new information
        context_updates = {
            "relevant_chunks": doc_context,
            "history": history + [(question, answer.text)]  # Update history
        }

        return ActionResult(
            return_value=answer.text,
            context_updates=context_updates
        )
    except Exception as e:
        print(f"Error in rag(): {e}")
        return ActionResult(
            return_value="An error occurred while processing your query.",
            context_updates={}
        )

def init(app: LLMRails):
    app.register_action(
        rag, 
        name="rag"
    )
    #    context_fn=lambda: {"index": index}  # Use context_fn for in-memory index
    #)
        #fn=lambda  messages, **kwargs: asyncio.run(rag({"index": index, "messages": messages, **kwargs}))  # Pass index here
    #)

#def init(app: LLMRails, index):  # Add index as a parameter
#    app.register_action(rag, name="rag", context_fn=lambda: {"index": index})
