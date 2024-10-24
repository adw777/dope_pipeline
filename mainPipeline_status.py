import os 
import numpy as np
import logging
from typing import List, Dict, Any
from bson import ObjectId
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http import models
from create_qdrant_collection import create_collections
from content import (
    fetch_urls_from_mongo,
    update_document_with_original_content,
    store_embeddings_in_qdrant,
    fetch_titles_and_urls_from_mongo,
    update_document_with_summary_and_keywords,
    store_summary_embedding,
    store_keyword_embeddings,
    update_contentCol_with_keywords
)
from extraction2 import extractContent
from docPartition import sortDocuments
from embeddings import getEmbeddings
from clustering import clusteringPipeline
from summarisation import summarizationPipeline
from keywordExtraction import keywordPipeline, generate_semantic_meaning_long_docs, generate_semantic_meaning_short_docs
import concurrent.futures

from loggingConfig import setupLogging
os.makedirs('logs', exist_ok=True)

setupLogging(defaultPath='logging.yaml')
logger = logging.getLogger(__name__)

logger.info("Pipeline started")
logger.debug("Debug logging is working")
logger.error("Error logging is working")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_and_store_original_content(db_name: str, collection_name: str, doc: Dict[str, Any]):
    try:
        url = doc.get('pdfUrl') or doc.get('pdfLink') or doc.get('pdfUrls') or doc.get('Link')
        if not url:
            logger.error(f"No URL found for document {doc['_id']}")
            return False

        content = extractContent(url)
        sorted_content = sortDocuments(content)
        
        # Update MongoDB with original content
        update_result = update_document_with_original_content(db_name, collection_name, doc['_id'], sorted_content['text'])
        if "error" in update_result:
            logger.error(f"Failed to update original content for document {doc['_id']}: {update_result['error']}")
            return False
        
        # Generate and store embeddings
        embeddings = getEmbeddings(sorted_content['text'])
        if len(embeddings) > 0:
            embedding = embeddings[0] if embeddings.ndim > 1 else embeddings
            qdrant_success = store_embeddings_in_qdrant(collection_name, ObjectId(doc['_id']), doc.get('title', ''), url, embedding)
            if not qdrant_success:
                logger.warning(f"Possible issue storing embeddings for document {doc['_id']} in Qdrant")
        else:
            logger.error(f"No embeddings generated for document {doc['_id']}")
            return False

        logger.info(f"Successfully processed and stored original content for document {doc['_id']}")
        return True, embeddings
    except Exception as e:
        logger.error(f"Error processing document {doc['_id']}: {str(e)}")
        return False

def keySum(content_embedding):

    content = list(content_embedding.keys())
    vectors = np.array([np.array(embed) for embed in content_embedding.values()])
    
    if len(content) > 2:
        mostSimilarChunks, outliers = clusteringPipeline(content,vectors)  # similar chunks, outliers
        summary = summarizationPipeline(mostSimilarChunks,outliers)  # summary
        chunks = mostSimilarChunks + outliers
        keywords = keywordPipeline(chunks,generate_semantic_meaning_long_docs)
    else:
        mostSimilarChunks = content
        outliers = []
        summary = summarizationPipeline(mostSimilarChunks,outliers)
        keywords = keywordPipeline(mostSimilarChunks,generate_semantic_meaning_short_docs)
    
    return summary, keywords

def generate_summary_and_keywords(db_name: str, collection_name: str, doc: Dict[str, Any], content_embeddings: np.ndarray = None):
    try:
        doc_id = doc['_id']
        original_content = doc['originalContent']

        # Create content_embedding dictionary for keySum
        content_embedding = {}
        if isinstance(original_content, list):
            if content_embeddings is None or len(content_embeddings) != len(original_content):
                # Only generate embeddings if not provided or length mismatch
                logger.warning(f"Generating new embeddings for document {doc_id} due to missing or mismatched embeddings")
                content_embeddings = getEmbeddings(original_content)

            for i, content in enumerate(original_content):
                content_embedding[content] = content_embeddings[i]
        else:
            if content_embeddings is None:
                # Only generate embeddings if not provided
                logger.warning(f"Generating new embeddings for document {doc_id} due to missing embeddings")
                content_embeddings = getEmbeddings(original_content)
            
            content_embedding[original_content] = content_embeddings if content_embeddings.ndim == 1 else content_embeddings[0]

        # Generate summary and keywords using keySum
        summary, keywords = keySum(content_embedding)

        # Update MongoDB with summary and keywords
        update_result = update_document_with_summary_and_keywords(db_name, collection_name, doc_id, summary, keywords)
        if "error" in update_result:
            logger.error(f"Failed to update summary and keywords for document {doc_id}: {update_result['error']}")
            return False
        
        # Update contentColA with keywords
        update_contentCol_with_keywords(collection_name, doc_id, keywords)
        
        # Generate and store embeddings for summary and keywords concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            summary_embedding_future = executor.submit(getEmbeddings, summary)
            keyword_embedding_future = executor.submit(getEmbeddings, keywords)
            
            summary_embeddings = summary_embedding_future.result()
            keyword_embeddings = keyword_embedding_future.result()
        
        if len(summary_embeddings) > 0 and len(keyword_embeddings) > 0:
            # Store summary embedding
            summary_success = store_summary_embedding(
                doc_id,
                doc.get('title', ''),
                doc.get('pdfUrl') or doc.get('pdfLink') or doc.get('pdfUrls') or doc.get('Link', ''),
                summary_embeddings,
                keywords,
                collection_name
            )
            if not summary_success:
                logger.warning(f"Possible issue storing summary embeddings for document {doc_id} in Qdrant")

            # Store keyword embeddings
            keywords_success = store_keyword_embeddings(
                doc_id,
                doc.get('title', ''),
                doc.get('pdfUrl') or doc.get('pdfLink') or doc.get('pdfUrls') or doc.get('Link', ''),
                keyword_embeddings,
                collection_name
            )
            if not keywords_success:
                logger.warning(f"Possible issue storing keyword embeddings for document {doc_id} in Qdrant")
            
            if summary_success and keywords_success:
                logger.info(f"Successfully stored summary and keyword embeddings for document {doc_id}")
            else:
                logger.warning(f"Partially succeeded in storing embeddings for document {doc_id}")
        else:
            logger.error(f"Failed to generate embeddings for summary or keywords for document {doc_id}")
            return False

        logger.info(f"Finished processing summary and keywords for document {doc_id}")
        return True
    except Exception as e:
        logger.error(f"Error processing summary and keywords for document {doc_id}: {str(e)}")
        return False

# processes mongo documents in batches!
def process_document(db_name: str, collection_name: str, doc: Dict[str, Any]):
    try:
        doc_id = doc['_id']
        client = MongoClient('mongodb+srv://doadmin:67K98DEUBAY0T214@lwai-mongo-c557243a.mongo.ondigitalocean.com/stale?authSource=admin&tls=true')
        db = client[db_name]
        collection = db[collection_name]

        # Validate document structure
        if not isinstance(doc.get('originalContent', ''), (str, list)):
            logger.error(f"Invalid originalContent format for document {doc_id}")
            return

        current_status = doc.get('status', 'pending')
        if current_status == "updated":
            logger.info(f"Document {doc_id} is already updated. Skipping.")
            return

        try:
            if current_status == "pending":
                success, embeddings = process_and_store_original_content(db_name, collection_name, doc)
                if success and embeddings is not None:
                    # Fetch updated document after processing original content
                    updated_doc = collection.find_one({"_id": doc_id})
                    if updated_doc and generate_summary_and_keywords(db_name, collection_name, updated_doc, embeddings):
                        collection.update_one({"_id": doc_id}, {"$set": {"status": "updated"}})
                        logger.info(f"Document {doc_id} fully processed")
                    else:
                        collection.update_one({"_id": doc_id}, {"$set": {"status": "parsed"}})
                        logger.warning(f"Document {doc_id} partially processed")
            elif current_status == "parsed":
                if generate_summary_and_keywords(db_name, collection_name, doc):
                    collection.update_one({"_id": doc_id}, {"$set": {"status": "updated"}})
                    logger.info(f"Document {doc_id} summary and keywords processed")

        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {str(e)}")
            
    finally:
        if 'client' in locals():
            client.close()

def process_single_collection(db_name: str, collection_name: str, limit: int = None):
    client = MongoClient('mongodb+srv://doadmin:67K98DEUBAY0T214@lwai-mongo-c557243a.mongo.ondigitalocean.com/stale?authSource=admin&tls=true')
    db = client[db_name]
    
    logging.info(f"Processing collection: {collection_name}")
    
    # Check if collection exists
    if db[collection_name].count_documents({}) == 0:
        logging.info(f"Collection {collection_name} is empty. Skipping.")
        client.close()
        return

    # Process documents in batches to avoid cursor timeout
    batch_size = 100
    processed_count = 0
    
    try:
        while True:
            # Query with batch limiting and sort to ensure consistent ordering
            query = {}
            cursor = db[collection_name].find(query).sort('_id', 1).skip(processed_count)
            if limit:
                cursor = cursor.limit(min(batch_size, limit - processed_count))
            else:
                cursor = cursor.limit(batch_size)

            # Convert cursor to list to avoid timeout during processing
            batch_docs = list(cursor)
            if not batch_docs:
                break

            # Process each document in the batch
            for doc in batch_docs:
                try:
                    process_document(db_name, collection_name, doc)
                except Exception as e:
                    logger.error(f"Error processing document {doc.get('_id', 'unknown')}: {str(e)}")
                    continue

            processed_count += len(batch_docs)
            logger.info(f"Processed {processed_count} documents in {collection_name}")

            if limit and processed_count >= limit:
                break

    except Exception as e:
        logger.error(f"Error processing collection {collection_name}: {str(e)}")
    finally:
        client.close()

def main_pipeline(mode: str = "all", db_name: str = "andhra", collection_name: str = None, limit: int = None):
    if mode == "single" and collection_name is None:
        logger.error("For single collection mode, collection_name must be provided.")
        return

    if mode == "single":
        process_single_collection(db_name, collection_name, limit)
    else:
        client = MongoClient('mongodb+srv://doadmin:67K98DEUBAY0T214@lwai-mongo-c557243a.mongo.ondigitalocean.com/stale?authSource=admin&tls=true')
        db = client[db_name]
        collections = db.list_collection_names()

        for coll_name in collections:
            process_single_collection(db_name, coll_name, limit)

        client.close()

if __name__ == "__main__":
    # Process all collections (original behavior)
    # main_pipeline(mode="all", limit=None)

    # To process a single collection
    main_pipeline(
        mode="single",
        db_name="andhra",
        collection_name="Tirupati_District",
        limit=None  # limit for number of documents to be processed
    )
    

"""
to serve vllm-
vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --max_model_len 60000 --api_key key-ar123
"""














