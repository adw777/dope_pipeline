# import urllib3
# from typing import Dict, Any, List
# from openai import OpenAI

# urllib3.disable_warnings()

# def modelRun(messages:List[Dict[str, str]]):
#     client = OpenAI(
#         base_url="http://localhost:8000/v1",
#         api_key="key-ar123")
    
#     completion = client.chat.completions.create(
#         model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
#         messages=messages)
        
#     return completion.choices[0].message.content 

import urllib3
from typing import Dict, Any, List
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from functools import partial

urllib3.disable_warnings()

def single_model_run(message: Dict[str, str], client: OpenAI) -> str:
    """
    Run a single model inference.
    
    Args:
        message: Single message to process
        client: OpenAI client instance
    Returns:
        str: Model response content
    """
    try:
        completion = client.chat.completions.create(
            model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
            messages=[message]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in single model run: {e}")
        return ""

def modelRun(messages: List[Dict[str, str]], batch_size: int = 10) -> List[str]:
    """
    Run model inference concurrently on multiple messages.
    
    Args:
        messages: List of messages to process
        batch_size: Number of concurrent requests to process (default: 10)
    Returns:
        List[str]: List of model responses
    """
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="key-ar123"
    )
    
    results = []
    
    # Process messages in batches
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        # Create a partial function with the client
        run_with_client = partial(single_model_run, client=client)
        
        # Submit all tasks and get futures
        futures = [executor.submit(run_with_client, message) for message in messages]
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing message: {e}")
                results.append("")
    
    return results