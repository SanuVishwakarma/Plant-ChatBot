from pathlib import Path
import json
from typing import List, Dict, Optional
import logging
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from langchain_core.documents import Document

from ragbase.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IngestionError(Exception):
    """Custom exception for ingestion errors"""
    pass

class JsonIngestor:
    def __init__(self):
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=Config.Model.EMBEDDINGS,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            self.semantic_splitter = SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type="interquartile"
            )
            
            self.recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2048,
                chunk_overlap=128,
                add_start_index=True,
            )
            
            logger.info("Successfully initialized JsonIngestor")
            
        except Exception as e:
            logger.error(f"Failed to initialize JsonIngestor: {str(e)}")
            raise IngestionError(f"Initialization failed: {str(e)}")

    def load_json_data(self, json_path: Path) -> List[Dict]:
        """
        Load and parse JSON data with error handling
        
        Args:
            json_path (Path): Path to the JSON file
            
        Returns:
            List[Dict]: List of parsed JSON objects
            
        Raises:
            IngestionError: If file reading or parsing fails
        """
        try:
            if not json_path.exists():
                raise FileNotFoundError(f"JSON file not found: {json_path}")
                
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            logger.info(f"Successfully loaded JSON data from {json_path}")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            raise IngestionError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading JSON data: {str(e)}")
            raise IngestionError(f"Failed to load JSON data: {str(e)}")

    def convert_to_documents(self, json_data: List[Dict]) -> List[Document]:
        """
        Convert JSON data to Document objects with enhanced metadata
        
        Args:
            json_data (List[Dict]): List of JSON objects
            
        Returns:
            List[Document]: List of processed Document objects
        """
        try:
            documents = []
            for item in json_data:
                # Validate required fields
                required_fields = ['title', 'description', 'price']
                missing_fields = [field for field in required_fields if field not in item]
                if missing_fields:
                    logger.warning(f"Missing required fields {missing_fields} in item: {item.get('title', 'Unknown')}")
                    continue

                # Create a more structured content string with explicit labels
                content_parts = []
                
                # Essential information
                content_parts.extend([
                    f"Title: {item.get('title', 'N/A')}",
                    f"Description: {item.get('description', 'N/A')}",
                    f"Price: ${item.get('price', 'N/A')}",
                ])
                
                # Optional pricing information
                if 'markedPrice' in item:
                    content_parts.append(f"Marked Price: ${item['markedPrice']}")
                if 'discountPercentage' in item:
                    content_parts.append(f"Discount: {item['discountPercentage']}%")
                
                # Product details
                optional_fields = [
                    ('quantity', 'Quantity Available'),
                    ('rating', 'Rating'),
                    ('reviewsCount', 'Reviews Count'),
                    ('size', 'Size', lambda x: ', '.join(x) if isinstance(x, list) else x),
                    ('sunlightRequirement', 'Sunlight Requirement'),
                    ('waterFrequency', 'Water Frequency'),
                    ('waterFrequencyDescription', 'Water Frequency Description'),
                    ('place', 'Location'),
                    ('growthRate', 'Growth Rate'),
                    ('benefits', 'Benefits', lambda x: ' '.join(x) if isinstance(x, list) else x),
                    ('category', 'Category'),
                    ('nutritionalNeeds', 'Nutritional Needs'),
                    ('seasonalAvailability', 'Seasonal Availability'),
                    ('propagationMethod', 'Propagation Method', lambda x: ' '.join(x) if isinstance(x, list) else x),
                    ('pestResistance', 'Pest Resistance'),
                    ('toxicityLevel', 'Toxicity Level')
                ]
                
                for field, label, transform_func in ((*f, lambda x: x) if len(f) == 2 else f for f in optional_fields):
                    if field in item and item[field]:
                        content_parts.append(f"{label}: {transform_func(item[field])}")
                
                # Tags and categories
                for tag_field in ['tag', 'plantTags', 'promotionTags', 'plantAccessories', 'plantCare']:
                    if tag_field in item and item[tag_field]:
                        value = item[tag_field]
                        if isinstance(value, list):
                            value = ', '.join(value)
                        content_parts.append(f"{tag_field}: {value}")
                
                content = "\n".join(content_parts)
                
                # Enhanced metadata
                metadata = {
                    "source": "product_data",
                    "title": item.get('title', 'Unknown'),
                    "price": item.get('price', 0),
                    "category": item.get('category', 'Unknown'),
                    "ingestion_timestamp": datetime.now().isoformat(),
                }
                
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            
            logger.info(f"Successfully converted {len(documents)} items to documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error converting JSON to documents: {str(e)}")
            raise IngestionError(f"Document conversion failed: {str(e)}")

    def ingest(self, json_path: Path) -> VectorStore:
        """
        Ingest JSON data into vector store with progress logging
        
        Args:
            json_path (Path): Path to the JSON file
            
        Returns:
            VectorStore: Initialized vector store with embedded documents
            
        Raises:
            IngestionError: If any step of the ingestion process fails
        """
        try:
            logger.info(f"Starting ingestion process for {json_path}")
            
            # Load JSON data
            json_data = self.load_json_data(json_path)
            logger.info(f"Loaded {len(json_data)} items from JSON")
            
            # Convert to documents
            initial_documents = self.convert_to_documents(json_data)
            logger.info(f"Converted to {len(initial_documents)} documents")
            
            # Combine document texts
            document_text = "\n".join(
                [doc.page_content for doc in initial_documents]
            )
            
            # Split documents using both splitters
            logger.info("Applying semantic chunking...")
            semantic_chunks = self.semantic_splitter.create_documents([document_text])
            logger.info(f"Created {len(semantic_chunks)} semantic chunks")
            
            logger.info("Applying recursive splitting...")
            final_chunks = self.recursive_splitter.split_documents(semantic_chunks)
            logger.info(f"Created {len(final_chunks)} final chunks")
            
            # Create vector store
            logger.info("Creating vector store...")
            vector_store = Qdrant.from_documents(
                documents=final_chunks,
                embedding=self.embeddings,
                path=Config.Path.DATABASE_DIR if not Config.Database.COLLECTION_PATH else None,
                url=Config.Database.COLLECTION_PATH if Config.Database.COLLECTION_PATH else None,
                collection_name=Config.Database.DOCUMENTS_COLLECTION,
            )
            
            logger.info("Successfully completed ingestion process")
            return vector_store
            
        except Exception as e:
            logger.error(f"Fatal error during ingestion: {str(e)}")
            raise IngestionError(f"Ingestion process failed: {str(e)}")

    def update_documents(self, json_path: Path, vector_store: Optional[VectorStore] = None) -> VectorStore:
        """
        Update existing documents or create new ones
        
        Args:
            json_path (Path): Path to the JSON file
            vector_store (Optional[VectorStore]): Existing vector store to update
            
        Returns:
            VectorStore: Updated vector store
        """
        try:
            if vector_store is None:
                return self.ingest(json_path)
            
            # Load and process new documents
            json_data = self.load_json_data(json_path)
            documents = self.convert_to_documents(json_data)
            
            # Update vector store
            vector_store.add_documents(documents)
            logger.info(f"Successfully updated vector store with {len(documents)} documents")
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error updating documents: {str(e)}")
            raise IngestionError(f"Document update failed: {str(e)}")