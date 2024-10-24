import concurrent.futures
from typing import List, Union, Tuple
from model import modelRun
import tiktoken
from prompts import summaryPrompt, combinePrompt, outlierPrompt, singleCallPrompt, singleCallKeyPrompt

def summaryCall(chunk: str) -> str:
    """
    Generates a summary for a given text chunk.
    Args:
        chunk (str): The text chunk to summarize.

    Returns:
        str: The generated summary.
    """
    try:
        formattedPrompt = summaryPrompt.format(text=chunk)
        prompt = [{"role": "user", "content": formattedPrompt}]
        result: str = modelRun(prompt)
        return result
    except Exception as e:
        print(f"Error in summaryCall: {e}")
        return ""
    
def summaryOutlier(chunk: str) -> str:
    """
    Generates a summary for a given text chunk.
    Args:
        chunk (str): The text chunk to summarize.

    Returns:
        str: The generated summary.
    """
    try:
        formattedPrompt = outlierPrompt.format(text=chunk)
        prompt = [{"role": "user", "content": formattedPrompt}]
        result: str = modelRun(prompt)
        return result
    except Exception as e:
        print(f"Error in summaryOutlier: {e}")
        return ""

def semanticSum(chunksArray: List[str]) -> List[str]:
    """
    Generates summaries for a list of text chunks concurrently.
    Args:
        chunksArray (List[str]): A list of text chunks to summarize.

    Returns:
        List[str]: A list of generated summaries.
    """
    summaries: List[str] = []
    keywords: List[str] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            summaries = list(executor.map(summaryCall, chunksArray))
            executor.shutdown(wait=True)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            keywords = list(executor.map(keyCall, chunksArray))
            executor.shutdown(wait=True)
            
        return summaries, keywords
    except Exception as e:
        print(f"Error in semanticSum: {e}")
        return summaries
    
def outlierSum(chunksArray: List[str]) -> List[str]:
    """
    Generates summaries for a list of text chunks concurrently.
    Args:
        chunksArray (List[str]): A list of text chunks to summarize.

    Returns:
        List[str]: A list of generated summaries.
    """
    summaries: List[str] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            summaries = list(executor.map(summaryOutlier, chunksArray))
            executor.shutdown(wait=True)
        return summaries
    except Exception as e:
        print(f"Error in semanticSum: {e}")
        return summaries

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text using tiktoken.
    
    Args:
        text (str): Text to count tokens for
    Returns:
        int: Number of tokens
    """
    try:
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return 0

# def combineSummaries(summaries: List[str]) -> Union[str, Tuple[str, str]]:
#     """
#     Combines a list of summaries into a single text.
#     If combined text exceeds 60k tokens, splits it into two parts.
    
#     Args:
#         summaries (List[str]): List of summaries to combine
#     Returns:
#         Union[str, Tuple[str, str]]: Either the combined summary if under token limit,
#                                     or a tuple of two parts if over limit
#     """
#     try:

#         # Handle case where input is a list of lists
#         flattened_summaries = []
#         for summary in summaries:
#             if isinstance(summary, list):
#                 # If summary is a list, join its elements
#                 flattened_summaries.extend(summary)
#             elif isinstance(summary, str):
#                 # If summary is a string, add it directly
#                 flattened_summaries.append(summary)
#             else:
#                 print(f"Warning: Unexpected type in summaries: {type(summary)}")
#                 continue

#         # Filter out empty strings and join
#         valid_summaries = [s for s in flattened_summaries if s and isinstance(s, str)]
#         if not valid_summaries:
#             return ""

#         combined_text = '\n\n'.join(summaries)
        
#         token_count = count_tokens(combined_text)
        
#         if token_count <= 60000:
#             return combined_text
            
#         # If over limit, split into two roughly equal parts
#         encoder = tiktoken.get_encoding("cl100k_base")
#         tokens = encoder.encode(combined_text)
#         mid_point = len(tokens) // 2
        
#         first_half = encoder.decode(tokens[:mid_point])
#         second_half = encoder.decode(tokens[mid_point:])
        
#         return (first_half, second_half)
        
#     except Exception as e:
#         print(f"Error in combineSummaries: {e}")
#         return ""

# def combineCall(summaries: List[str], outliers:List[str]) -> str:
#     """
#     Generates a verbose summary from a list of summaries.
#     Args:
#         summaries (List[str]): A list of summaries to combine and summarize.

#     Returns:
#         str: The generated verbose summary.
#     """
#     try:
#         formattedSummary = combineSummaries(summaries)
#         formattedOutliers = combineSummaries(outliers) if outliers else ""

#         # Handle case where either summaries or outliers are split due to token limit
#         if isinstance(formatted_summary, tuple) or isinstance(formatted_outliers, tuple):
#             # Process first half
#             first_summary = formatted_summary[0] if isinstance(formatted_summary, tuple) else formatted_summary
#             first_outliers = formatted_outliers[0] if isinstance(formatted_outliers, tuple) else formatted_outliers
            
#             first_prompt = combinePrompt.format(text=first_summary, outliers=first_outliers)
#             first_result = modelRun([{"role": "user", "content": first_prompt}])
            
#             # Process second half if it exists
#             if isinstance(formatted_summary, tuple) or isinstance(formatted_outliers, tuple):
#                 second_summary = formatted_summary[1] if isinstance(formatted_summary, tuple) else formatted_summary
#                 second_outliers = formatted_outliers[1] if isinstance(formatted_outliers, tuple) else formatted_outliers
                
#                 second_prompt = combinePrompt.format(text=second_summary, outliers=second_outliers)
#                 second_result = modelRun([{"role": "user", "content": second_prompt}])
                
#                 # Combine both results with another summarization pass
#                 combined_text = f"{first_result}\n\n{second_result}"
#                 final_prompt = combinePrompt.format(text=combined_text, outliers="")
#                 return modelRun([{"role": "user", "content": final_prompt}])
            
#             return first_result

#         formattedPrompt = combinePrompt.format(text=formattedSummary, outliers=formattedOutliers)
#         prompt = [{"role": "user", "content": formattedPrompt}]
#         result: str = modelRun(prompt)
#         return result

#     except Exception as e:
#         print(f"Error in combineCall: {e}")
#         return ""

# def summarizationPipelineOrg(chunks: List[str], outliers:List[str]) -> str:
#     """
#     The complete summarization pipeline.
#     Args:
#         chunks (List[str]): A list of text chunks to summarize.

#     Returns:
#         str: The final verbose summary.
#     """
#     try:
#         if(outliers==[]):
#             summaries = semanticSum(chunks)
#         elif(chunks==[]):
#             outliers: List[str] = outlierSum(outliers)
#         else:
#             summaries = semanticSum(chunks)
#             outliers: List[str] = outlierSum(outliers)
#         result: str = combineCall(summaries, outliers)
#         return result
#     except Exception as e:
#         print(f"Error in summarizationPipeline: {e}")
#         return ""
    
# def summarizer(chunks: List[str])->str:
#     try: 
#         formattedSummary = combineSummaries(chunks)
#         formattedPrompt = singleCallPrompt.format(text=formattedSummary)
#         keyprompt=singleCallKeyPrompt.format(text=formattedSummary)
#         result=modelRun(formattedPrompt, max_tokens=2048)
#         keywords=modelRun(keyprompt, max_tokens=1024)
#         return result, keywords
#     except Exception as e:
#         print(f"Error in summarizer: {e}")
#         return ""
    
# def summarizationPipeline(chunks: List[str], outliers: List[str]) -> str:
#     # try:
#     #     summaries,outlierSum = list(),list()
#     #     if(outliers==[]):
#     #         for chunk in chunks:
#     #             summary = summaryCall(chunk)
#     #             summaries.append(summary)
#     #         outlierSum = ['']
#     #     elif(chunks==[]):
#     #         for outlier in outliers:
#     #             summaryOut = summaryOutlier(outlier)
#     #             outlierSum.append(summaryOut)
#     #         summaries=['']
#     #     else:
#     #         for chunk in chunks:
#     #             summary = summaryCall(chunk)
#     #             summaries.append(summary)
#     #         for outlier in outliers:
#     #             summaryOut = summaryOutlier(outlier)
#     #             outlierSum.append(summaryOut)
#     #     result: str = combineCall(summaries, outlierSum)
#     #     return result
#     # except Exception as e:
#     #     print(f"Error in summarizationPipeline: {e}")
#     #     return ""

#     try:
#         summaries = []
#         outlier_summaries = []
        
#         # Process chunks if they exist
#         if chunks:
#             for chunk in chunks:
#                 if chunk:  # Skip empty chunks
#                     summary = summaryCall(chunk)
#                     if summary:
#                         summaries.append(summary)
        
#         # Process outliers if they exist
#         if outliers:
#             for outlier in outliers:
#                 if outlier:  # Skip empty outliers
#                     outlier_summary = summaryOutlier(outlier)
#                     if outlier_summary:
#                         outlier_summaries.append(outlier_summary)
        
#         # If no valid summaries were generated, return empty string
#         if not summaries and not outlier_summaries:
#             return ""
        
#         # Combine all summaries
#         result = combineCall(summaries, outlier_summaries)
#         return result
#     except Exception as e:
#         print(f"Error in summarizationPipeline: {e}")
#         return ""
        

def flatten_nested_content(content: any) -> List[str]:
    """
    Recursively flatten nested lists and convert content to strings.
    
    Args:
        content: Any type of content (str, list, or nested lists)
    Returns:
        List[str]: Flattened list of strings
    """
    flattened = []
    
    if isinstance(content, str):
        return [content]
    elif isinstance(content, list):
        for item in content:
            flattened.extend(flatten_nested_content(item))
    else:
        # Convert any other type to string
        return [str(content)]
        
    return flattened

def combineSummaries(summaries: Union[List[str], List[List[str]], str]) -> Union[str, Tuple[str, str]]:
    """
    Combines summaries into a single text, handling nested lists and different types.
    If combined text exceeds 60k tokens, splits it into two parts.
    
    Args:
        summaries: List of summaries or nested lists of summaries
    Returns:
        Union[str, Tuple[str, str]]: Either combined summary if under token limit,
                                    or tuple of two parts if over limit
    """
    try:
        # Print debug information
        print(f"Input summaries type: {type(summaries)}")
        if isinstance(summaries, list) and summaries:
            print(f"First element type: {type(summaries[0])}")
        
        # Handle single string input
        if isinstance(summaries, str):
            return summaries
            
        # Flatten and clean the summaries
        flattened_summaries = flatten_nested_content(summaries)
        
        # Filter out empty or None values
        valid_summaries = [s for s in flattened_summaries if s and isinstance(s, str)]
        
        if not valid_summaries:
            print("Warning: No valid summaries found after filtering")
            return ""
            
        # Join the valid summaries
        combined_text = '\n\n'.join(valid_summaries)
        token_count = count_tokens(combined_text)
        
        if token_count <= 60000:
            return combined_text
            
        # If over limit, split into two roughly equal parts
        encoder = tiktoken.get_encoding("cl100k_base")
        tokens = encoder.encode(combined_text)
        mid_point = len(tokens) // 2
        
        first_half = encoder.decode(tokens[:mid_point])
        second_half = encoder.decode(tokens[mid_point:])
        
        return (first_half, second_half)
        
    except Exception as e:
        print(f"Error in combineSummaries: {e}")
        print(f"Input summaries: {summaries}")
        return ""

def summarizationPipeline(chunks: List[str], outliers: List[str]) -> str:
    """
    The complete summarization pipeline.
    Args:
        chunks: List of text chunks to summarize
        outliers: List of outlier chunks

    Returns:
        str: The final summary
    """
    try:
        summaries = []
        outlier_summaries = []
        
        # Debug print
        print(f"Input chunks type: {type(chunks)}, first chunk type: {type(chunks[0]) if chunks else 'No chunks'}")
        
        # Process main chunks
        if chunks:
            if isinstance(chunks, list):
                for chunk in chunks:
                    if chunk:  # Skip empty chunks
                        if isinstance(chunk, list):
                            # Handle nested lists
                            for subchunk in chunk:
                                summary = summaryCall(subchunk)
                                if summary:
                                    summaries.append(summary)
                        else:
                            summary = summaryCall(chunk)
                            if summary:
                                summaries.append(summary)
            else:
                summary = summaryCall(chunks)
                if summary:
                    summaries.append(summary)
        
        # Process outliers
        if outliers:
            if isinstance(outliers, list):
                for outlier in outliers:
                    if outlier:  # Skip empty outliers
                        if isinstance(outlier, list):
                            # Handle nested lists
                            for suboutlier in outlier:
                                outlier_summary = summaryOutlier(suboutlier)
                                if outlier_summary:
                                    outlier_summaries.append(outlier_summary)
                        else:
                            outlier_summary = summaryOutlier(outlier)
                            if outlier_summary:
                                outlier_summaries.append(outlier_summary)
            else:
                outlier_summary = summaryOutlier(outliers)
                if outlier_summary:
                    outlier_summaries.append(outlier_summary)
        
        # Debug print
        print(f"Generated summaries type: {type(summaries)}, length: {len(summaries)}")
        print(f"Generated outlier summaries type: {type(outlier_summaries)}, length: {len(outlier_summaries)}")
        
        # If no valid summaries were generated, return empty string
        if not summaries and not outlier_summaries:
            return ""
        
        # Combine all summaries
        result = combineCall(summaries, outlier_summaries)
        return result
    except Exception as e:
        print(f"Error in summarizationPipeline: {e}")
        print(f"Debug - chunks: {type(chunks)}, outliers: {type(outliers)}")
        return ""

def combineCall(summaries: List[str], outliers: List[str]) -> str:
    """
    Generates a verbose summary from lists of summaries.
    Args:
        summaries: List of summaries to combine and summarize
        outliers: List of outlier summaries

    Returns:
        str: The generated verbose summary
    """
    try:
        # Debug print
        print(f"combineCall input - summaries type: {type(summaries)}, outliers type: {type(outliers)}")
        
        # Get formatted summaries
        formatted_summary = combineSummaries(summaries)
        formatted_outliers = combineSummaries(outliers) if outliers else ""
        
        # Debug print
        print(f"Formatted summary type: {type(formatted_summary)}")
        print(f"Formatted outliers type: {type(formatted_outliers)}")
        
        # Handle case where either summaries or outliers are split due to token limit
        if isinstance(formatted_summary, tuple) or isinstance(formatted_outliers, tuple):
            first_summary = formatted_summary[0] if isinstance(formatted_summary, tuple) else formatted_summary
            first_outliers = formatted_outliers[0] if isinstance(formatted_outliers, tuple) else formatted_outliers
            
            first_prompt = combinePrompt.format(text=first_summary, outliers=first_outliers)
            first_result = modelRun([{"role": "user", "content": first_prompt}])
            
            if isinstance(formatted_summary, tuple) or isinstance(formatted_outliers, tuple):
                second_summary = formatted_summary[1] if isinstance(formatted_summary, tuple) else formatted_summary
                second_outliers = formatted_outliers[1] if isinstance(formatted_outliers, tuple) else formatted_outliers
                
                second_prompt = combinePrompt.format(text=second_summary, outliers=second_outliers)
                second_result = modelRun([{"role": "user", "content": second_prompt}])
                
                combined_text = f"{first_result}\n\n{second_result}"
                final_prompt = combinePrompt.format(text=combined_text, outliers="")
                return modelRun([{"role": "user", "content": final_prompt}])
            
            return first_result
        
        # Handle normal case where no splitting was needed
        formatted_prompt = combinePrompt.format(text=formatted_summary, outliers=formatted_outliers)
        return modelRun([{"role": "user", "content": formatted_prompt}])
    
    except Exception as e:
        print(f"Error in combineCall: {e}")
        print(f"Debug - summaries: {type(summaries)}, outliers: {type(outliers)}")
        return ""