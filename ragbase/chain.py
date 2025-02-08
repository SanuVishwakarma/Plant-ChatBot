import re
from operator import itemgetter
from typing import List, AsyncGenerator, Union, Dict, Any

from langchain.schema.runnable import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStoreRetriever

from ragbase.config import Config
from ragbase.session_history import get_session_history

SYSTEM_PROMPT = """
You are a helpful plant shop assistant. Use the provided context to answer customer questions about plants.
When answering questions about prices, quantities, or other numeric values, be sure to state the exact number from the context.
If a question asks about a specific number (price, quantity, rating, etc.), provide the precise value.
If you cannot find the answer in the context, simply say that you don't have that information.

Context:
{context}

Always format prices with the $ symbol and be precise with numbers.
"""

def format_documents(documents: List[Document]) -> str:
    texts = []
    for doc in documents:
        content = doc.page_content.strip()
        content = re.sub(r'[{}\[\]]', '', content)
        texts.append(content)
        texts.append("---")
    return "\n".join(texts)

def create_chain(llm: BaseLanguageModel, retriever: VectorStoreRetriever) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question")
            | retriever.with_config({"run_name": "context_retriever"})
            | format_documents
        )
        | prompt
        | llm
    )

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    ).with_config({"run_name": "chain_answer"})

async def ask_question(
    chain: Runnable, 
    question: str, 
    session_id: str
) -> AsyncGenerator[Union[str, List[Document]], None]:
    try:
        async for event in chain.astream_events(
            {"question": question},
            config={
                "configurable": {"session_id": session_id},
            },
            version="v2",
            include_names=["context_retriever", "chain_answer"],
        ):
            event_type = event["event"]
            if event_type == "on_retriever_end":
                yield event["data"]["output"]
            if event_type == "on_chain_stream":
                yield event["data"]["chunk"].content
    except Exception as e:
        yield f"Error processing question: {str(e)}"