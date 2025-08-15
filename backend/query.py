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
        model="mistral:7b-instruct-q4_K_M",
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
    """
    Perform a RAG query on the PGVectorStore.
    """
    chat_instance = chat_instances[chat_id]
    chat = chat_instance["chat"]

    prompt = chat_instance["messages"]

    # retriever_multi_query = MultiQueryRetriever.from_llm(
    #     retriever=vector_store.as_retriever(),
    #     llm=chat,
    # )

    # # Perform the retrieval
    # docs = retriever_multi_query.invoke(query)

    # retriever = create_history_aware_retriever(
    #     llm=chat,
    #     retriever=vector_store.as_retriever(),
    #     prompt=ChatPromptTemplate.from_template(REPHRASE_PROMPT),
    # )

    # Perform the retrieval
    # docs = retriever.invoke({
    #     "input": query,
    #     "chat_history": prompt,
    # })
    rephrased_query = chat.invoke(PromptTemplate.from_template(REPHRASE_PROMPT).format(
        chat_history=prompt,
        input=query
    ), think=False)
    print(f"Rephrased query: {rephrased_query.content}")

    docs = retriever.invoke(rephrased_query.content, search_type="similarity")
    print("Length of documents:", len(docs))

    # Take the top 50 documents
    # docs = docs[:50]

    context = "\n---\n".join([result.page_content for result in docs]) if docs else "No relevant documents found."
    for doc in docs:
        print(f"Document: {doc.metadata['source']}: with metadata {doc.metadata}")
    
    # print(context)

    prompt.append(
        {"role": "user", "content": ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
            context=context,
            question=query
        )}
    )

    # disable thinking 
    response = chat.invoke(prompt, think=False)

    sources = {doc.metadata["source"]: doc.metadata for doc in docs}

    sources_formatted = []
    for source, metadata in sources.items():
        match metadata["source_type"]:
            case "youtube":
                sources_formatted.append(f'[{metadata["title"]}]({metadata["webpage_url"]}) by [{metadata["author"]}](https://www.youtube.com/channel/{metadata["channel_id"]})')
            case "markdown":
                sources_formatted.append(f'[{metadata["source"]}]({metadata["source"]})')
            case "website":
                sources_formatted.append(f'[{metadata["source"]}]({metadata["source"]})')
    
    formatted_response = {
        "response": response.content,
        "sources": sources_formatted
    }

    prompt.append(
        {"role": "assistant", "content": response.content}
    )

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


