import os
import logging
import numpy as np
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from loggingConfig import setupLogging
from sentence_transformers import SentenceTransformer

# Load environment variables and setup logging
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPEN_API_KEY"))
emModel = os.environ.get("EM_MODEL")

setupLogging()
logger = logging.getLogger(__name__)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)


# Load the model based on the environment variable
if emModel == "openSource":
    model = SentenceTransformer(
        model_name_or_path="intfloat/e5-mistral-7b-instruct", 
        trust_remote_code=True, 
        device="cuda", 
        cache_folder="Models", 
        prompts={"Clustering": "Identify the most pertinent chunks in legal passage here to summarize. "}
    )
elif emModel == "text-embedding-3-large":
    pass
else:
    logger.error(f"Unknown Model Specified: {emModel}")
    raise ValueError(f"Unkown Model Specfied: {emModel}")    
# Uncomment the following block if you want to use a different model
# elif emModel == "openSource":
#     model = SentenceTransformer(
#         model_name_or_path="Alibaba-NLP/gte-Qwen2-7B-instruct", 
#         trust_remote_code=True, 
#         device="cuda", 
#         cache_folder="Models", 
#         prompts=["Clustering: Identify the most pertinent chunks in this legal document"]
#     )

def getEmbeddings(documents: List[str], batch_size: int = 1024, emModel: str = emModel) -> List[List[float]] | List[str]:
    """
    Generates embeddings for a list of documents using the specified model.

    Args:
        documents (List[str]): A list of documents (strings) to generate embeddings for.
        batch_size (int): The number of documents to process in a batch. Default is 1024.
        emModel (str): The model to use for generating embeddings.

    Returns:
        List[List[float]]: A list of embeddings, each embedding being a list of floats.
    """
    try:
        document_embeddings: List[List[float]] = []
        if emModel == "text-embedding-3-large":
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                response = client.embeddings.create(input=batch, model=emModel, dimensions=1024)
                for item in response.data:
                    document_embeddings.append(item.embedding)
        elif emModel == "openSource":
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_embeddings = model.encode(sentences=batch, device="cuda", batch_size=batch_size)
                document_embeddings.extend(batch_embeddings)
        else:
            logger.error(f"Unknown model specified: {emModel}")
            return  np.array([])
        """
        return  np.array(document_embeddings)
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}", exc_info=True)
        return np.array([])
        """
    # Ensure each embedding is 1024 dimensions
        document_embeddings = [embedding[:1024] for embedding in document_embeddings]
        return  np.array(document_embeddings)
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}", exc_info=True)
        return np.array([])