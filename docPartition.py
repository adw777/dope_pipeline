import logging
import tiktoken
from typing import List
from tiktoken.core import Encoding
from loggingConfig import setupLogging
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Setup logging using the provided configuration
setupLogging()
logger = logging.getLogger(__name__)

# Initialize tokenizer
tokenizer: Encoding = tiktoken.get_encoding("cl100k_base")

# Initialize text splitters with specific configurations
midSplit = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=768, chunk_overlap=128
) 

highSplit = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=1536, chunk_overlap=512
)

def countTokens(text: str) -> int:
    """
    Counts the number of tokens in a given text using the tokenizer.

    Args:
        text (str): The input text to be tokenized.

    Returns:
        int: The number of tokens in the text.
    """
    return len(tokenizer.encode(text))

def sortDocuments(text: str) -> dict[str:any]:
    """
    Sorts documents based on the number of tokens and creates chunks accordingly.

    Args:
        text (str): The input text to be chunked.

    Returns:
        List[str]: A list of text chunks.
    """
    try:
        numTokens: int = countTokens(text)
        document={}
        if numTokens < 2536:
            document['text']=text
            document['level']=1
            return document
        elif numTokens < 11264:
            textChunks = midSplit.split_text(text)
            document['text']=textChunks
            document['level']=2
            return document
        else:
            textChunks = highSplit.split_text(text)
            document['text']=textChunks
            document['level']=3
            return document
    except Exception as e:
        logger.error(f"Error during document sorting: {e}", exc_info=True)
        return [text]