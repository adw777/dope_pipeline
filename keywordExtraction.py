# import ast
# from model import modelRun
# from prompts import keywordPromptSD,correctKeywords, keywordPromptLD
# #from groq import Groq
# from openai import OpenAI

# def generate_semantic_meaning_short_docs(document_chunk):
#     try:
#         formattedPrompt = keywordPromptSD.format(text=document_chunk)
#         prompt = [{'role': "user", 'content':formattedPrompt}]
#         result = modelRun(prompt)
#         return result
#     except Exception as e:
#         print(f"Error in Keyword Generation: {e}")

# def generate_semantic_meaning_long_docs(document_chunk):
#     try:
#         formattedPrompt = keywordPromptLD.format(text=document_chunk)
#         prompt = [{'role': "user", 'content':formattedPrompt}]
#         result = modelRun(prompt)
#         return result
#     except Exception as e:
#         print(f"Error in Keyword Generation: {e}")

# '''
# def corrected_list(keywords):
#     try:
#         Prompt = correctKeywords.format(text=keywords)
#         prompt = [{'role': "user", 'content':Prompt}]
#         result = modelRun(prompt)
#         return result
#     except Exception as e:
#         print(f"Error in correcting Keywords: {e}")
# '''

# def convert_to_list(s):
#     result = []
#     current = ''
#     in_string = False
#     i = 0
#     while i < len(s):
#         c = s[i]
#         if c == '"':
#             in_string = not in_string  # Toggle the in_string state
#             if not in_string:  # Closing quote found, add the current element to the list
#                 result.append(current.strip())
#                 current = ''
#         elif c == ',' and not in_string:
#             if current.strip():
#                 result.append(current.strip())  # Add the current element outside quotes
#                 current = ''
#         elif c in '[]' and not in_string:
#             pass  # Ignore brackets outside of strings
#         else:
#             current += c  # Accumulate characters into current string
#         i += 1
#     if current.strip():  # Append last element if exists
#         result.append(current.strip())
#     return result

# def corrected_list(keywords):
#     client = OpenAI(
#         base_url="http://localhost:8000/v1",
#         api_key="key-ar123")
#     chat_completion = client.chat.completions.create(
#         messages=[
#             {
#                 "role":"system",
#                 "content":'''
#                 You are an expert Indian legal analyst with extensive experience in legal document review and keyword extraction. You have complete understanding of Indian legal academia. Your task is to compile a precise list of keywords from the extracted keywords and context provided for a legal document. The keywords are extracted from various passages of the legal document The final keywords should capture the main legal topics, statutes, case laws, and important legal principles discussed throughout the document.
#                 Please refine this list by:
#                 1. Removing redundant or repetitive terms while preserving the most specific and meaningful legal terms.
#                 2. Treating distinct legal terms, such as "Article 338" and "Article 366", as different.
#                 3. For similar terms (e.g., "The Gazette of India" and "TheGAZETTE OF INDIA EXTRAORDINARY"), retain only one of the most representative terms.
#                 4. Return only a Python list of the refined terms. Avoid adding any terms not present in the original list.
#                 5. Do not provide any additional comments or explanations. Just return the refined list.
#                 6. Say nothing else. For example, don't say: "Here are the keywords present in the document"
# '''
#             },
#             {
#                 "role":"user",
#                 "content":keywords
#             }
#         ],
#         model = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
#     )
#     return chat_completion.choices[0].message.content


# def keywordPipeline(chunks,function):
#     try:
#         keywords = list()
#         for chunk in chunks:
#             keys = function(chunk)
#             try:
#                 keys = ast.literal_eval(keys)
#                 keywords.extend(keys)
#             except:
#                 keys = convert_to_list(keys)
#                 keywords.extend(keys)
#         keywords = ', '.join(map(str,set(keywords)))
#         #print(keywords)
#         result = corrected_list(keywords)
#         try:
#             result = ast.literal_eval(result)
#         except:
#             result = convert_to_list(result)    
#         return result
#     except Exception as e:
#         print(f"Error in keywordPipeline: {e}")
#         return ""
    

import ast
import concurrent.futures
from typing import List, Dict
from model import modelRun
from prompts import keywordPromptSD, correctKeywords, keywordPromptLD
from openai import OpenAI

def process_chunk_short(chunk: str) -> str:
    """Process a single chunk using short document prompt."""
    try:
        prompt = [{'role': "user", 'content': keywordPromptSD.format(text=chunk)}]
        return modelRun(prompt)
    except Exception as e:
        print(f"Error in process_chunk_short: {e}")
        return ""

def process_chunk_long(chunk: str) -> str:
    """Process a single chunk using long document prompt."""
    try:
        prompt = [{'role': "user", 'content': keywordPromptLD.format(text=chunk)}]
        return modelRun(prompt)
    except Exception as e:
        print(f"Error in process_chunk_long: {e}")
        return ""

def convert_to_list(s):
    result = []
    current = ''
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == '"':
            in_string = not in_string  # Toggle the in_string state
            if not in_string:  # Closing quote found, add the current element to the list
                result.append(current.strip())
                current = ''
        elif c == ',' and not in_string:
            if current.strip():
                result.append(current.strip())  # Add the current element outside quotes
                current = ''
        elif c in '[]' and not in_string:
            pass  # Ignore brackets outside of strings
        else:
            current += c  # Accumulate characters into current string
        i += 1
    if current.strip():  # Append last element if exists
        result.append(current.strip())
    return result

def corrected_list(keywords: str) -> str:
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="key-ar123"
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": '''
                You are an expert Indian legal analyst with extensive experience in legal document review and keyword extraction. You have complete understanding of Indian legal academia. Your task is to compile a precise list of keywords from the extracted keywords and context provided for a legal document. The keywords are extracted from various passages of the legal document The final keywords should capture the main legal topics, statutes, case laws, and important legal principles discussed throughout the document.
                Please refine this list by:
                1. Removing redundant or repetitive terms while preserving the most specific and meaningful legal terms.
                2. Treating distinct legal terms, such as "Article 338" and "Article 366", as different.
                3. For similar terms (e.g., "The Gazette of India" and "TheGAZETTE OF INDIA EXTRAORDINARY"), retain only one of the most representative terms.
                4. Return only a Python list of the refined terms. Avoid adding any terms not present in the original list.
                5. Do not provide any additional comments or explanations. Just return the refined list.
                6. Say nothing else. For example, don't say: "Here are the keywords present in the document"
                '''
            },
            {
                "role": "user",
                "content": keywords
            }
        ],
        model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    )
    return chat_completion.choices[0].message.content

def keywordPipeline(chunks: List[str], function) -> List[str]:
    """
    Process chunks concurrently to extract keywords using concurrent.futures.map().
    
    Args:
        chunks: List of text chunks to process
        function: Function to determine processing type (short/long doc)
    Returns:
        List of processed keywords
    """
    try:
        # Determine which processing function to use
        process_func = process_chunk_short if function == generate_semantic_meaning_short_docs else process_chunk_long
        
        # Process chunks concurrently using ThreadPoolExecutor
        keywords = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Use map to process chunks concurrently
            results = list(executor.map(process_func, chunks))
            
            # Process results
            for result in results:
                try:
                    keys = ast.literal_eval(result)
                    keywords.extend(keys)
                except:
                    keys = convert_to_list(result)
                    keywords.extend(keys)
        
        # Create final keyword string
        keywords = ', '.join(map(str, set(keywords)))
        
        # Get corrected list
        result = corrected_list(keywords)
        
        # Parse final result
        try:
            result = ast.literal_eval(result)
        except:
            result = convert_to_list(result)
            
        return result
    except Exception as e:
        print(f"Error in keywordPipeline: {e}")
        return []

# Keep these as simple wrappers for backward compatibility
def generate_semantic_meaning_short_docs(document_chunk: str) -> str:
    return process_chunk_short(document_chunk)

def generate_semantic_meaning_long_docs(document_chunk: str) -> str:
    return process_chunk_long(document_chunk)


if __name__ == "__main__":
    # Sample legal text chunks for testing
    test_chunks = [
        """
        The Supreme Court of India, in its landmark judgment dated 15th December 2023, 
        addressing the interpretation of Article 370 of the Indian Constitution, held that 
        the President's power under Article 356 extends to taking irreversible actions. 
        The Court examined the constitutional validity of various amendments and their impact 
        on the special status of Jammu and Kashmir.
        """,
        
        """
        Under Section 124A of the Indian Penal Code, sedition remains a contentious issue. 
        The Delhi High Court, referring to the case of Kedar Nath Singh v. State of Bihar, 
        emphasized that criticism of government actions does not constitute sedition unless 
        it incites violence or creates public disorder.
        """,
        
        """
        The Right to Information Act, 2005 mandates public authorities to provide information 
        within 30 days of request. The Central Information Commission, in a recent order, 
        clarified that public interest overshadows privacy concerns when dealing with 
        matters of environmental impact and public health.
        """,
        
        """
        The Competition Commission of India, exercising powers under Section 3 of the 
        Competition Act, 2002, imposed penalties on companies engaging in cartelization. 
        The order highlighted the importance of maintaining fair market practices and 
        preventing anti-competitive agreements.
        """
    ]
    
    print("\nTesting Keyword Extraction Pipeline...")
    print("-" * 50)
    
    # Test with short document processing
    print("\nTesting Short Document Processing:")
    try:
        short_keywords = keywordPipeline(test_chunks, generate_semantic_meaning_short_docs)
        print("Extracted Keywords (Short Doc):")
        for i, keyword in enumerate(short_keywords, 1):
            print(f"{i}. {keyword}")
    except Exception as e:
        print(f"Error in short document processing: {e}")
    
    print("\n" + "-" * 50)
    
    # Test with long document processing
    print("\nTesting Long Document Processing:")
    try:
        long_keywords = keywordPipeline(test_chunks, generate_semantic_meaning_long_docs)
        print("Extracted Keywords (Long Doc):")
        for i, keyword in enumerate(long_keywords, 1):
            print(f"{i}. {keyword}")
    except Exception as e:
        print(f"Error in long document processing: {e}")
    
    # Test concurrent processing with a larger set of chunks
    print("\n" + "-" * 50)
    print("\nTesting Concurrent Processing with Extended Chunks:")
    
    # Create more test chunks by repeating the existing ones
    extended_chunks = test_chunks * 3  # Creates 12 chunks total
    
    try:
        print(f"\nProcessing {len(extended_chunks)} chunks concurrently...")
        concurrent_keywords = keywordPipeline(extended_chunks, generate_semantic_meaning_short_docs)
        print("\nExtracted Keywords (Concurrent Processing):")
        for i, keyword in enumerate(concurrent_keywords, 1):
            print(f"{i}. {keyword}")
    except Exception as e:
        print(f"Error in concurrent processing: {e}")

    print("\n" + "-" * 50)
    print("Testing Complete!")