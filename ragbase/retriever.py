from typing import Optional

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_filter import LLMChainFilter
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_qdrant import Qdrant

from ragbase.config import Config
from ragbase.model import create_embeddings, create_reranker

class RetrieverError(Exception):
    """Custom exception for retriever errors"""
    pass

def create_retriever(
    llm: BaseLanguageModel, 
    vector_store: Optional[VectorStore] = None
) -> VectorStoreRetriever:
    try:
        if not vector_store:
            vector_store = Qdrant.from_existing_collection(
                embedding=create_embeddings(),
                collection_name=Config.Database.DOCUMENTS_COLLECTION,
                path=Config.Path.DATABASE_DIR if not Config.Database.COLLECTION_PATH else None,
                url=Config.Database.COLLECTION_PATH if Config.Database.COLLECTION_PATH else None,
            )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": Config.Retriever.TOP_K}
        )

        if Config.Retriever.USE_RERANKER:
            try:
                retriever = ContextualCompressionRetriever(
                    base_compressor=create_reranker(),
                    base_retriever=retriever
                )
            except Exception as e:
                print(f"Warning: Reranker initialization failed: {str(e)}")

        if Config.Retriever.USE_CHAIN_FILTER:
            try:
                retriever = ContextualCompressionRetriever(
                    base_compressor=LLMChainFilter.from_llm(llm),
                    base_retriever=retriever
                )
            except Exception as e:
                print(f"Warning: Chain filter initialization failed: {str(e)}")

        return retriever
        
    except Exception as e:
        raise RetrieverError(f"Failed to create retriever: {str(e)}")