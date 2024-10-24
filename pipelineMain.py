import os 
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from bson import ObjectId
from pymongo import MongoClient
from qdrant_client import QdrantClient
# from qdrant_client.http import models
import concurrent.futures
from dataclasses import dataclass
from content import (
    store_embeddings_in_qdrant,
    store_summary_embedding,
    store_keyword_embeddings,
)
from extraction2 import extractContent
from docPartition import sortDocuments
from embeddings import getEmbeddings
from clustering import clusteringPipeline
from summarisation import summarizationPipeline
from keywordExtraction import keywordPipeline, generate_semantic_meaning_long_docs, generate_semantic_meaning_short_docs

from loggingConfig import setupLogging
os.makedirs('logs', exist_ok=True)
setupLogging(defaultPath='logging.yaml')
logger = logging.getLogger(__name__)

@dataclass
class DocumentState:
    """Class to hold document processing state"""
    doc_id: ObjectId
    title: str
    url: str
    original_content: str
    original_embeddings: Optional[np.ndarray] = None
    summary: Optional[str] = None
    keywords: Optional[List[str]] = None
    summary_embeddings: Optional[np.ndarray] = None
    keyword_embeddings: Optional[np.ndarray] = None
    status: str = "pending"

class DocumentProcessor:
    def __init__(self, db_name: str, collection_name: str):
        self.db_name = db_name
        self.collection_name = collection_name
        self.mongo_client = MongoClient('mongodb+srv://doadmin:67K98DEUBAY0T214@lwai-mongo-c557243a.mongo.ondigitalocean.com/stale?authSource=admin&tls=true')
        self.qdrant_client = QdrantClient(url='http://64.227.154.249:6333/', port=6333)
        self.db = self.mongo_client[db_name]
        self.collection = self.db[collection_name]

    def parse_document(self, doc: Dict[str, Any]) -> Optional[DocumentState]:
        """Parse document and extract content"""
        try:
            doc_id = doc['_id']
            url = doc.get('pdfUrl') or doc.get('pdfLink') or doc.get('pdfUrls') or doc.get('Link')
            if not url:
                logger.error(f"No URL found for document {doc_id}")
                return None

            content = extractContent(url)
            if not content:
                logger.error(f"Failed to extract content for document {doc_id}")
                return None

            sorted_content = sortDocuments(content)
            
            return DocumentState(
                doc_id=doc_id,
                title=doc.get('title', ''),
                url=url,
                original_content=sorted_content['text']
            )
        except Exception as e:
            logger.error(f"Error parsing document {doc.get('_id', 'unknown')}: {str(e)}")
            return None

    def process_content(self, doc_state: DocumentState) -> bool:
        """Process document content - generate embeddings, summary, and keywords"""
        try:
            # Generate embeddings for original content
            original_embeddings = getEmbeddings(doc_state.original_content)
            if len(original_embeddings) == 0:
                logger.error(f"Failed to generate embeddings for document {doc_state.doc_id}")
                return False
            
            doc_state.original_embeddings = original_embeddings

            # Create content_embedding dictionary for clustering
            content_embedding = {}
            if isinstance(doc_state.original_content, list):
                for i, content in enumerate(doc_state.original_content):
                    content_embedding[content] = original_embeddings[i]
            else:
                content_embedding[doc_state.original_content] = original_embeddings

            # Generate summary and keywords
            summary, keywords = self.generate_summary_keywords(content_embedding)
            if not summary or not keywords:
                logger.error(f"Failed to generate summary or keywords for document {doc_state.doc_id}")
                return False

            doc_state.summary = summary
            doc_state.keywords = keywords

            # Generate embeddings for summary and keywords concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                summary_future = executor.submit(getEmbeddings, summary)
                keywords_future = executor.submit(getEmbeddings, keywords)
                
                doc_state.summary_embeddings = summary_future.result()
                doc_state.keyword_embeddings = keywords_future.result()

            if len(doc_state.summary_embeddings) == 0 or len(doc_state.keyword_embeddings) == 0:
                logger.error(f"Failed to generate embeddings for summary or keywords of document {doc_state.doc_id}")
                return False

            return True
        except Exception as e:
            logger.error(f"Error processing content for document {doc_state.doc_id}: {str(e)}")
            return False

    def generate_summary_keywords(self, content_embedding: Dict[str, np.ndarray]) -> Tuple[str, List[str]]:
        """Generate summary and keywords from content"""
        try:
            content = list(content_embedding.keys())
            vectors = np.array([np.array(embed) for embed in content_embedding.values()])
            
            if len(content) > 2:
                mostSimilarChunks, outliers = clusteringPipeline(content, vectors)
                summary = summarizationPipeline(mostSimilarChunks, outliers)
                chunks = mostSimilarChunks + outliers
                keywords = keywordPipeline(chunks, generate_semantic_meaning_long_docs)
            else:
                summary = summarizationPipeline(content, [])
                keywords = keywordPipeline(content, generate_semantic_meaning_short_docs)
            
            return summary, keywords
        except Exception as e:
            logger.error(f"Error generating summary and keywords: {str(e)}")
            return None, None

    def update_mongo(self, doc_state: DocumentState) -> bool:
        """Update MongoDB with processed document state"""
        try:
            update_data = {
                "originalContent": doc_state.original_content,
                "summary": doc_state.summary,
                "keywords": doc_state.keywords,
                "status": "updated"
            }
            
            result = self.collection.update_one(
                {"_id": doc_state.doc_id},
                {"$set": update_data}
            )
            
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating MongoDB for document {doc_state.doc_id}: {str(e)}")
            return False

    def update_qdrant(self, doc_state: DocumentState) -> bool:
        """Update Qdrant with all embeddings"""
        try:
            # Store original content embeddings
            content_success = store_embeddings_in_qdrant(
                self.collection_name,
                doc_state.doc_id,
                doc_state.title,
                doc_state.url,
                doc_state.original_embeddings
            )

            # Store summary embeddings
            summary_success = store_summary_embedding(
                doc_state.doc_id,
                doc_state.title,
                doc_state.url,
                doc_state.summary_embeddings,
                doc_state.keywords,
                self.collection_name
            )

            # Store keyword embeddings
            keywords_success = store_keyword_embeddings(
                doc_state.doc_id,
                doc_state.title,
                doc_state.url,
                doc_state.keyword_embeddings,
                self.collection_name
            )

            return all([content_success, summary_success, keywords_success])
        except Exception as e:
            logger.error(f"Error updating Qdrant for document {doc_state.doc_id}: {str(e)}")
            return False

    def process_batch(self, batch_docs: List[Dict[str, Any]]) -> None:
        """Process a batch of documents"""
        for doc in batch_docs:
            try:
                # Skip already processed documents
                if doc.get('status') == "updated":
                    logger.info(f"Document {doc['_id']} already processed. Skipping.")
                    continue

                # Parse document
                doc_state = self.parse_document(doc)
                if not doc_state:
                    continue

                # Process content
                if not self.process_content(doc_state):
                    continue

                # Update MongoDB
                if not self.update_mongo(doc_state):
                    continue

                # Update Qdrant
                if not self.update_qdrant(doc_state):
                    continue

                logger.info(f"Successfully processed document {doc_state.doc_id}")

            except Exception as e:
                logger.error(f"Error processing document {doc.get('_id', 'unknown')}: {str(e)}")
                continue

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mongo_client.close()

def process_collection(db_name: str, collection_name: str, batch_size: int = 100, limit: Optional[int] = None):
    """Process entire collection in batches"""
    logger.info(f"Starting processing of collection: {collection_name}")
    
    try:
        with DocumentProcessor(db_name, collection_name) as processor:
            # Check if collection exists
            if processor.collection.count_documents({}) == 0:
                logger.info(f"Collection {collection_name} is empty. Skipping.")
                return

            processed_count = 0
            while True:
                # Query for batch
                cursor = processor.collection.find({}).sort('_id', 1).skip(processed_count)
                if limit:
                    cursor = cursor.limit(min(batch_size, limit - processed_count))
                else:
                    cursor = cursor.limit(batch_size)

                batch_docs = list(cursor)
                if not batch_docs:
                    break

                # Process batch
                processor.process_batch(batch_docs)
                
                processed_count += len(batch_docs)
                logger.info(f"Processed {processed_count} documents in {collection_name}")

                if limit and processed_count >= limit:
                    break

    except Exception as e:
        logger.error(f"Error processing collection {collection_name}: {str(e)}")

def main_pipeline(mode: str = "all", db_name: str = "andhra", collection_name: str = None, limit: int = None):
    """Main pipeline entry point"""
    logger.info("Starting main pipeline")
    
    try:
        if mode == "single":
            if not collection_name:
                logger.error("Collection name required for single mode")
                return
            process_collection(db_name, collection_name, limit=limit)
        else:
            client = MongoClient('mongodb+srv://doadmin:67K98DEUBAY0T214@lwai-mongo-c557243a.mongo.ondigitalocean.com/stale?authSource=admin&tls=true')
            db = client[db_name]
            collections = db.list_collection_names()
            client.close()

            for coll_name in collections:
                process_collection(db_name, coll_name, limit=limit)

        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")

if __name__ == "__main__":
    main_pipeline(
        mode="single",
        db_name="andhra",
        collection_name="Krishna_District",
        limit=None #None for all docs
    )
