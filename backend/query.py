from langchain_postgres import PGEngine, PGVectorStore
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import OllamaEmbeddings, ChatOllama
import json
import uuid
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.storage import SQLStore
from langchain.storage._lc_store import create_kv_docstore
from os import getenv
# from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank, ModelType
# from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
# from rank_llm.rerank.listwise.zephyr_reranker import ZephyrReranker


# ------- prompt guard (Lllama Guard 3) ----------

from langchain_ollama import ChatOllama




# -- connect to DB -- 

CONNECTION_STRING = getenv("PG_CONNECTION_STRING", "")
if not CONNECTION_STRING:
    raise ValueError("PG_CONNECTION_STRING environment variable is not set.")

COLLECTION_NAME = "documents"
VECTOR_SIZE = 768  # Adjust based on the model's output vector size
DOCUMENT_STORE_NAMESPACE = "full_documents"

OLLAMA_BASE_URL = getenv("OLLAMA_BASE_URL", "http://localhost:11434")

chat_instances = {}

REPHRASE_PROMPT = '''
Task: Given a multi-turn conversation and a follow-up user question:

- Rewrite the follow-up as a clear, brief, and standalone question suitable for retrieving relevant documents from a vector database.
- Extract a concise list of the most relevant key concepts or phrases from the rewritten question for use in vector similarity search.
- Use the context from the full conversation to preserve intent and necessary background.
- The rewritten question should not reference the conversation explicitly (e.g., avoid “as mentioned before”).
- Ensure the standalone question includes all important entities, topics, and context implied in the follow-up.
- Focus on technical terms, specific entities, and concepts relevant to the question.
- Avoid unnecessary details or overly complex language.
- Extract only essential technical terms, entities, and concepts—avoid stopwords, vague verbs, or general filler words.
- Do not include any personal opinions or interpretations.

Chat History:
{chat_history}

Follow-up Question:
{input}
'''

PROMPT_TEMPLATE = '''
---

{context}

---
Answer the question based on the above context: {question}
'''

def get_vector_store():
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model="nomic-embed-text"
    )

    engine = PGEngine.from_connection_string(url=CONNECTION_STRING)

    vector_store = PGVectorStore.create_sync(
        engine=engine,
        embedding_service=embeddings,
        table_name=COLLECTION_NAME,
        k=3
    )

    return vector_store

def get_retriever(vector_store: PGVectorStore) -> ParentDocumentRetriever:
    sql_store = SQLStore(
        namespace=DOCUMENT_STORE_NAMESPACE,
        db_url=CONNECTION_STRING
    )
    doc_store = create_kv_docstore(sql_store)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=200,
        length_function=len,
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=doc_store,
        child_splitter=child_splitter,
        # parent_splitter=parent_splitter
    )

    # compressor = RankLLMRerank(
    #     top_n=3,
    #     client=ZephyrReranker(device="cpu"),
    #     model="zephyr",
    # )
    # compression_retriever = ContextualCompressionRetriever(
    #     base_retriever=retriever,
    #     base_compressor=compressor,
    #     return_source_documents=True
    # )
    # del compressor

    return retriever

def create_rag_chat() -> uuid.UUID:
    """
    Create a RAG chat instance using the Ollama model.
    """
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    chat = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model = os.getenv("LLM_MODEL", "gemma3") ,
        temperature=0.1,  # Adjust temperature for more deterministic responses
        prompt=prompt,
    )

    # Generate a unique ID for the chat instance
    chat_id = str(uuid.uuid4())
    chat_instances[chat_id] = {
        "chat": chat,
        "messages": [
            {
            "role": "system",
            "content": "You are a helpful assistant for University of Virginia Research Computing (UVA RC). You will answer questions concisely and accurately based on the provided context, which may include content from the UVA RC YouTube channel and UVA RC Teaching Markdowns.\n\n- If the question is related to Research Computing but not covered in the context, respond with: 'I'm not sure based on the provided information.'\n- If the question is unrelated to Research Computing or computing in general, respond with: 'I'm here to help with UVA Research Computing-related topics. Let me know if you have a relevant question.'\n- Do not invent answers. Base your responses only on the context and reliable background knowledge.\n- Be succinct, clear, and helpful in all responses."
            }
        ],
    }
    
    return chat_id

def rag_query(chat_id: uuid.UUID, retriever: ParentDocumentRetriever, query: str) -> str:

    print("\n----- NEW QUERY -----", flush=True)
    print("User asked:", query, flush=True)

    # -------------------------------
    # SAFETY CHECK
    # -------------------------------

    GUARD_PROMPT = f"""
You are Llama Guard 3, a safety classifier.
Respond ONLY with JSON in this format:

{{
  "allowed": true/false,
  "category": "safe" | "violence" | "self-harm" | "hate" | "sexual" | "crime" | "weapons" | "drugs" | "other",
  "explanation": "<short explanation>"
}}

Classify the following user message:

"{query}"
"""

    guard_model = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model="llama-guard3",
        temperature=0,
    )

    guard_raw = guard_model.invoke(guard_prompt, think=False)
    print("Llama Guard raw output:", guard_raw.content, flush=True)

    try:
        guard_result = json.loads(guard_raw.content)
    except:
        guard_result = {
            "allowed": False,
            "category": "error",
            "explanation": "Could not parse Llama Guard output"
        }

    print("Safety Check Result:", guard_result, flush=True)

    if not guard_result.get("allowed", False):
        return {
            "response": f"⚠️ {guard_result['explanation']}",
            "sources": []
        }


    # load chat instance
    chat_instance = chat_instances.get(chat_id)
    if chat_instance is None:
        return {"response": "Chat session expired. Refresh and try again.", "sources": []}

    # -----------------------------------------------------------
    # 2. Proceed with normal RAG pipeline
    # -----------------------------------------------------------

    chat = chat_instance["chat"]
    messages = chat_instance["messages"]

    # Rephrase w/ conversation history
    rephrased = chat.invoke(
        PromptTemplate.from_template(REPHRASE_PROMPT).format(
            chat_history=messages,
            input=query
        ),
        think=False
    ).content

    print(f"Rephrased Query: {rephrased}")

    # Retrieve documents
    docs = retriever.invoke(rephrased, search_type="similarity")
    print(f"Documents retrieved: {len(docs)}")

    context = "\n---\n".join([d.page_content for d in docs]) if docs else "No relevant documents found."

    # Build final prompt for RAG answer
    final_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context,
        question=query
    )

    messages.append({"role": "user", "content": final_prompt})

    # Run model (no "thinking")
    response = chat.invoke(messages, think=False)

    # --- Format sources ---
    sources = {doc.metadata["source"]: doc.metadata for doc in docs}
    sources_formatted = []
    for source, metadata in sources.items():
        if metadata["source_type"] == "youtube":
            sources_formatted.append(
                f'[{metadata["title"]}]({metadata["webpage_url"]})'
            )
        else:
            sources_formatted.append(f'[{metadata["source"]}]({metadata["source"]})')

    formatted_response = {
        "response": response.content,
        "sources": sources_formatted
    }

    # Append assistant message for chat history
    messages.append({"role": "assistant", "content": response.content})

    return formatted_response


def main():
    vector_store = get_vector_store()

    retriever = get_retriever(vector_store)

    rag_chat = create_rag_chat()

    while True:
        query_text = input("Enter your query: ")

        response = rag_query(rag_chat, retriever, query_text)

        formatted_response = json.dumps(response, indent=2)

        print(formatted_response)


if __name__ == "__main__":
    main()


