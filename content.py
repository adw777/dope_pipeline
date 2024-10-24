from pymongo import MongoClient
from bson import ObjectId
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

lookupTable: dict[str, dict[str, str]] = {
    'version': '1.0',
    'collections': {
        'eastgodavaris': 'C01',
        'wgods': 'C02',
        'Prakasham_District': 'C03',
        'ntrs': 'C04',
        'Chitoor_District': 'C05',
        'apgovs': 'C06',
        'kakinadas': 'C07',
        'sitharamanrajus': 'C08',
        'ankapallis': 'C09',
        'bapatlas': 'C10',
        'Tirupati_District': 'C11',
        'visakhapatnams': 'C12',
        'YSR_District': 'C13',
        'Krishna_District': 'C14',
        'nandyals': 'C15',
        'srikakulams': 'C16',
        'andhraGov': 'C17',
        'Sri_Potti_Sriramulu_Nellore_District': 'C18',
        'gunturs': 'C19',
        'Eluru_District': 'C20',
        'kurnool_district': 'C21',
        'srisathyasai_District': 'C22',
        'annamayyas': 'C23',
        'Parvathipuram_Manyam_District': 'C24',
        'konaseemas': 'C25',
        'Vizianagaram_District': 'C26',
        'Ananthapuramu_District': 'C27'
    }
}

def fetch_urls_from_mongo(db_name, collection_name, query, limit=None):
    # fetches title and pdfUrls from mongo db
    try:
        mongo_url = 'mongodb+srv://doadmin:67K98DEUBAY0T214@lwai-mongo-c557243a.mongo.ondigitalocean.com/stale?authSource=admin&tls=true'
        client = MongoClient(mongo_url)
        db = client[db_name]
        collection = db[collection_name]

        documents = collection.find(query, {"title": 1, "pdfUrl": 1})

        if limit is not None:
            documents = documents.limit(limit)

        result = [{"_id": str(doc["_id"]), "title": doc.get("title"), "url": doc.get("pdfUrl")} for doc in documents]

        client.close()

        return result

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def update_document_with_original_content(db_name, collection_name, object_id, original_content):
    # stores original_content in mongo db
    """
    Updates a document in the database with the provided original content.

    Parameters:
    - db_name (str): The name of the database.
    - collection_name (str): The name of the collection.
    - object_id (str): The ID of the document to update.
    - original_content (str): The original content of the document.

    Returns:
    - dict: A message indicating the update status.
    """

    mongo_url = 'mongodb+srv://doadmin:67K98DEUBAY0T214@lwai-mongo-c557243a.mongo.ondigitalocean.com/stale?authSource=admin&tls=true'
    client = MongoClient(mongo_url)
    db = client[db_name]
    collection = db[collection_name]
    
    try:
        document_id = ObjectId(object_id)
        
        update_field = {
            "originalContent": original_content
        }

        result = collection.update_one(
            {"_id": document_id},
            {"$set": update_field}
        )

        if result.matched_count == 0:
            return {"error": "Document not found"}
        if result.modified_count == 1:
            return {"message": "Document updated successfully with original content"}
        else:
            return {"message": "No changes made to the document"}

    except Exception as e:
        print("Error updating document:")
        return {"error": str(e)}

    finally:
        client.close()

def generate_point_id(doc_id: ObjectId, collection_name: str, lookupTable: dict[str, dict[str, str]]) -> str:
    # generating point id to store embeddings in qdrant
    try:
        chunkCode="C0000"
        colCode = lookupTable['collections'].get(collection_name, "C00") # C00 is default colCode
        return f"{chunkCode}{colCode}{str(doc_id)}"
    except KeyError as e:
        logger.warning(f"Collection name {collection_name} not found in lookupTable. Using default code C00.")
        return f"C0000C00{str(doc_id)}"

def store_embeddings_in_qdrant(collection_name: str, doc_id: ObjectId, title: str, url: str, embeddings: list):
    # storing original_content embeddings in qdrant
    try:
        qdrant_client = QdrantClient(url='http://64.227.154.249:6333/', port=6333)

        point_id = generate_point_id(doc_id, collection_name, lookupTable)

        # Handle different embedding shapes and ensure 1024 dimensions
        if isinstance(embeddings, np.ndarray):
            if embeddings.ndim > 1:
                # If we have multiple embeddings, take the mean
                embeddings = np.mean(embeddings, axis=0)
            embeddings = embeddings.flatten()
            
            # Ensure exactly 1024 dimensions
            if len(embeddings) > 1024:
                embeddings = embeddings[:1024]
            elif len(embeddings) < 1024:
                logger.error(f"Embedding dimension too small: {len(embeddings)}")
                return False
        else:
            embeddings = np.array(embeddings).flatten()
            if len(embeddings) != 1024:
                logger.error(f"Invalid embedding dimension: {len(embeddings)}")
                return False

        # Convert to list for Qdrant
        embeddings_list = embeddings.tolist()

        point = models.PointStruct(
            id=point_id,
            vector=embeddings_list,
            payload={
                "title": title,
                "url": url,
            }
        )

        operation_result = qdrant_client.upsert(
            collection_name="contentColA",
            points=[point]
        )

        return True
    except Exception as e:
        logger.error(f"Error storing embeddings in Qdrant: {e}")
        return False

def fetch_titles_and_urls_from_mongo(db_name, collection_name, query, limit=None):
    # fetching original_content from mongo with title and url
    try:
        mongo_url = 'mongodb+srv://doadmin:67K98DEUBAY0T214@lwai-mongo-c557243a.mongo.ondigitalocean.com/stale?authSource=admin&tls=true'
        client = MongoClient(mongo_url)
        db = client[db_name]
        collection = db[collection_name]

        documents = collection.find(query, {"_id": 1, "title": 1, "pdfUrl": 1, "originalContent": 1})

        if limit is not None:
            documents = documents.limit(limit)

        result = [{"id": doc.get("_id"), "title": doc.get("title"), "url": doc.get("pdfUrl"), "originalContent": doc.get("originalContent",[])} for doc in documents]

        client.close()

        return result

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def update_document_with_summary_and_keywords(db_name, collection_name, object_id, summary, keywords):
    # update mongo collection with summary and keyword
    """
    Updates a document in the database with the provided summary and keywords.

    Parameters:
    - object_id (str): The ID of the document to update.
    - summary (str): The summarized content of the document.
    - keywords (list): List of keywords extracted from the document.

    Returns:
    - dict: A message indicating the update status.
    """

    mongo_url = 'mongodb+srv://doadmin:67K98DEUBAY0T214@lwai-mongo-c557243a.mongo.ondigitalocean.com/stale?authSource=admin&tls=true'
    client = MongoClient(mongo_url)
    db = client[db_name]
    collection = db[collection_name]
    
    try:
        document_id = ObjectId(object_id)
    
        update_fields = {
            "summary": summary,
            "keywords": keywords
        }

        result = collection.update_one(
            {"_id": document_id},
            {"$set": update_fields}
        )

        if result.matched_count == 0:
            return {"error": "Document not found"}
        if result.modified_count == 1:
            return {"message": "Document updated successfully"}
        else:
            return {"message": "No changes made to the document"}

    except Exception as e:
        print("Error updating document:")
        return {"error": str(e)}

def generate_point_id_for_summary_and_keywords(doc_id: ObjectId, collection_name: str, lookupTable: dict[str, dict[str, str]]) -> str:
    # create point id for summary and keywords
    try:
        chunkCode="C9999"
        colCode = lookupTable['collections'].get(collection_name, "C00") # C00 is the default colCode
        return f"{chunkCode}{colCode}{str(doc_id)}"
    except KeyError as e:
        logger.warning(f"Collection name {collection_name} not found in lookupTable. Using default code C00.")
        return f"C9999C00{str(doc_id)}"    

def store_summary_embedding(doc_id: ObjectId, title: str, url: str, summary_embeddings: list, keywords: list, collection_name: str):
    try:
        qdrant_client = QdrantClient(url='http://64.227.154.249:6333/', port=6333)

        # Process embeddings to ensure correct format
        if isinstance(summary_embeddings, np.ndarray):
            if summary_embeddings.ndim > 1:
                # If we have multiple embeddings, take the mean
                summary_embeddings = np.mean(summary_embeddings, axis=0)
            summary_embeddings = summary_embeddings.flatten()
            
            # Ensure exactly 1024 dimensions
            if len(summary_embeddings) > 1024:
                summary_embeddings = summary_embeddings[:1024]
            elif len(summary_embeddings) < 1024:
                logger.error(f"Summary embedding dimension too small: {len(summary_embeddings)}")
                return False
        else:
            summary_embeddings = np.array(summary_embeddings).flatten()
            if len(summary_embeddings) != 1024:
                logger.error(f"Invalid summary embedding dimension: {len(summary_embeddings)}")
                return False

        # Convert to list for Qdrant
        embeddings_list = summary_embeddings.tolist()

        point_id = generate_point_id_for_summary_and_keywords(doc_id, collection_name, lookupTable)
        point = models.PointStruct(
            id=point_id,
            vector=embeddings_list,
            payload={
                "title": title,
                "url": url,
                "keywords": keywords
            }
        )

        operation_result = qdrant_client.upsert(
            collection_name="summaryColA",
            points=[point]
        )

        return True
    except Exception as e:
        logger.error(f"Error storing summary embeddings: {e}")
        return False
    
def store_keyword_embeddings(doc_id: ObjectId, title: str, url: str, keyword_embeddings: list, collection_name: str,):
    try:
        qdrant_client = QdrantClient(url='http://64.227.154.249:6333/', port=6333)

        # Process embeddings to ensure correct format
        if isinstance(keyword_embeddings, np.ndarray):
            if keyword_embeddings.ndim > 1:
                # If we have multiple embeddings, take the mean
                keyword_embeddings = np.mean(keyword_embeddings, axis=0)
            keyword_embeddings = keyword_embeddings.flatten()
            
            # Ensure exactly 1024 dimensions
            if len(keyword_embeddings) > 1024:
                keyword_embeddings = keyword_embeddings[:1024]
            elif len(keyword_embeddings) < 1024:
                logger.error(f"Keyword embedding dimension too small: {len(keyword_embeddings)}")
                return False
        else:
            keyword_embeddings = np.array(keyword_embeddings).flatten()
            if len(keyword_embeddings) != 1024:
                logger.error(f"Invalid keyword embedding dimension: {len(keyword_embeddings)}")
                return False

        # Convert to list for Qdrant
        embeddings_list = keyword_embeddings.tolist()

        point_id = generate_point_id_for_summary_and_keywords(doc_id, collection_name, lookupTable)
        point = models.PointStruct(
            id=point_id,
            vector=embeddings_list,
            payload={
                "title": title,
                "url": url
            }
        )

        operation_result = qdrant_client.upsert(
            collection_name="keywordColA",
            points=[point]
        )

        return True
    except Exception as e:
        logger.error(f"Error storing keyword embeddings: {e}")
        return False

def update_contentCol_with_keywords(collection_name: str, doc_id: ObjectId, keywords: list):
    # adds keywords as payload to the qdrant points of contentCol
    try:
        qdrant_client = QdrantClient(url='http://64.227.154.249:6333/', port=6333)
        point_id = generate_point_id(doc_id, collection_name, lookupTable)
        operation_result = qdrant_client.set_payload(
            collection_name="contentColA",
            payload={"keywords": keywords},
            points=[point_id]
        )
        """
        if operation_result.status == "ok":
            logger.info(f"Successfully added keywords as payload for document {doc_id} in Qdrant contentColA")
            return True
        else:
            logger.error(f"Failed to add keywords as payload for document {doc_id} in Qdrant contentColA")
            return False
        """
    except Exception as e:
        logger.error(f"Error adding keywords to the payload: {str(e)}")
        return False