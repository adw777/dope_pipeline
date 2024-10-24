summaryPrompt = """
You are an experienced legal analyst with a deep understanding of Indian law. Your task is to read and analyze a passage from an Indian legal document provided below. The passage will be enclosed in triple backticks (```). Based on this passage, you are to produce a comprehensive summary that accurately captures the key points, implications, and nuances of the legal text.

Your summary should:

First go through the entire passage to understand the objective and the purpose of the document. Try to understand the key points and salient features of the passage. Then, write a summary that:
1. Be at least three paragraphs long.
2. Clearly explain the salient features discussed in the passage.
3. Highlight any significant legal principles, rules, or legislation.
4. Address any potential implications or applications of the passage.
5. Back any point, implication, or discussion with information about the associated law or legislation if mentioned in the text.
6. Provide a precise and thorough overview, ensuring the reader fully understands the content and context of the legal text.

Passage:
```{text}```

FULL SUMMARY:

"""

outlierPrompt = """
You are an experienced legal analyst with a deep understanding of Indian law. Your task is to read and analyze a passage from an Indian legal document provided below. The passage will be enclosed in triple backticks (```). Based on this passage, you are to produce a concise and relevant summary that accurately captures the essential points, implications, and nuances of the legal text, without including unnecessary details or clutter.

Your summary should:
1. Be clear and concise, typically one to two paragraphs long.
2. Clearly explain the main points discussed in the passage.
3. Highlight any significant legal principles or rules.
4. Address any potential implications or applications of the passage.
5. Focus on the most relevant information, ensuring the reader understands the key content and context of the legal text.
6. Back any point, implication, or discussion with information about the associated law or legislation if mentioned in the text.

Passage:
```{text}```

CONCISE SUMMARY:

"""


combinePrompt = """
You are an expert legal analyst specializing in Indian law. You will be given a series of summaries derived from an Indian legal document, enclosed in triple backticks (```). These summaries include both the most important passages and outliers. Your task is to synthesize these summaries into a single, cohesive, and detailed summary that accurately reflects all the important and pertinent points covered in the original document.

Read the provided summaries carefully to understand the key points and outliers. Based on this information, understand the theme of the document and the salient features of the legal text.
Your final summary should:
- Create a precise summary in the following format,
- Integrate all key points from the provided summaries into a cohesive narrative.
- Clearly explain the main legal principles, rules, and implications discussed in the document.
- Highlight any recurring themes or significant details.
- Address any potential implications or applications of the document's content, back any point, implication, or discussion with information about the associated law or legislation if mentioned in the text.
- Ensure that the reader can achieve a complete understanding of the document's content and context from your summary.
- Be thorough and precise, leaving no important detail unaddressed.
- Avoid repetition of information or paragraphs, while ensuring that all important points are mentioned.
- Include only one conclusion in the end, summarsing all the content

Summaries of the most important passages:
```{text}```
Series of Outliers:
```{outliers}```

VERBOSE SUMMARY:

"""

singleCallPrompt = """
You are an experienced legal analyst with a deep understanding of Indian law. Your task is to read and analyze a legal document from an Indian legal academia provided below. The document will be enclosed in triple backticks (```). Based on this document, you are to produce a comprehensive summary that accurately captures the key points, implications, and nuances of the legal text.

Your summary should:

1. Be at least four paragraphs long.
2. Clearly explain the main points discussed in the document.
3. Highlight any significant legal principles, rules, or legislation.
4. Address any potential implications or applications of the passage.
5. Back any point, implication, or discussion with information about the associated law or legislation if mentioned in the text.
6. Provide a precise and thorough overview, ensuring the reader fully understands the content and context of the legal text.

document:
```{text}```

FULL SUMMARY:
"""

keywordPromptSD = """
You are an expert Indian legal analyst with extensive experience in legal document review and keyword extraction. Your task is to extract the most relevant keywords from a given legal passage from an Indian legal document.  The passage will be enclosed in triple backticks (```). Focus on identifying words and phrases that capture the main legal topics, statutes, case laws, and important Indian legal principles discussed in the text.

Follow these steps:

1. Carefully read the entire legal passage to understand its context, main points, and legal arguments.
2. Identify and list the main legal topics, statutes, and principles discussed.
3. Highlight specific keywords and phrases that hold significant relevance to the legal content.
4. Ensure to extract terms only in the legal context, avoid generic terms. If generic terms are required, they should be accompanied by a term in a legal context.
5. Return only a Python list of up to 5 unique terms, avoiding redundancy. 
6. Ensure each term is enclosed in double quotes, like this: "term". 
7. Output only the Python list and nothing else. Do not include any additional explanations or introductions., 
8. say nothing else. For example, don't say: "Here are the keywords present in the document"
9. Ensure the keywords are relevant and provide a comprehensive overview of the document's legal context.

Passage:
```{text}```

Extracted Keywords:

Ensure the list covers all possible keywords that can be extracted from the passage.
"""


keywordPromptLD = """
You are an expert Indian legal analyst with extensive experience in legal document review and keyword extraction. Your task is to extract the most relevant keywords from a given legal passage from an Indian legal document.  The passage will be enclosed in triple backticks (```). Focus on identifying words and phrases that capture the main legal topics, statutes, case laws, and important Indian legal principles discussed in the text.

Follow these steps:

1. Carefully read the entire legal passage to understand its context, main points, and legal arguments.
2. Identify and list the main legal topics, statutes, and principles discussed.
3. Highlight specific keywords and phrases that hold significant relevance to the legal content.
4. Ensure to extract terms only in the legal context, avoid generic terms. If generic terms are required, they should be accompanied by a term in a legal context.
5. Return only a Python list of up to 2 unique terms, avoiding redundancy. 
6. Ensure each term is enclosed in double quotes, like this: "term". 
7. Output only the Python list and nothing else. Do not include any additional explanations or introductions., 
8. say nothing else. For example, don't say: "Here are the keywords present in the document"
9. Ensure the keywords are relevant and provide a comprehensive overview of the document's legal context.

Passage:
```{text}```

Extracted Keywords:

Ensure the list covers all possible keywords that can be extracted from the passage.
"""

singleCallKeyPrompt = """
You are an expert Indian legal analyst with extensive experience in legal document review and keyword extraction. Your task is to extract the most relevant keywords from a given legal document from an Indian legal Academia.  The document will be enclosed in triple backticks (```). Focus on identifying words and phrases that capture the main legal topics, statutes, case laws, and important legal principles discussed in the text.

Follow these steps:

1.Carefully read the entire legal document to understand its context, main points, and legal arguments.
2.Identify and list the main legal topics, statutes, and principles discussed.
3.Highlight specific keywords or phrases that are frequently mentioned or hold significant relevance to the legal content, such as names of laws, sections, legal terms, and 4/important case references.
4.Ensure the keywords are relevant and provide a comprehensive overview of the document's legal context.
5.Compile a comprehensive list of the most relevant keywords that provide an accurate overview of the document's legal context.

Passage:
```{text}```

Extracted Keywords:

Ensure the list covers all possible keywords that can be extracted from the passage.
"""


correctKeywords="""
You are an expert Indian legal analyst with extensive experience in legal document review and keyword extraction. You have complete understanding of Indian legal academia. Your task is to compile a precise list of keywords from the extracted keywords and context provided for a legal document. The keywords are extracted from various passages of the legal document. They will be enclosed in triple backticks (```). The final keywords should capture the main legal topics, statutes, case laws, and important legal principles discussed throughout the document.

Follow these steps:

1. Review the provided lists of keywords extracted from various passages within the legal document.
2. Assess the relevance of each keyword based on its frequency and significance within the document through context provided.
3. Do not keep any duplicates in the final list.
4. Treating distinct legal terms, such as "Article 338" and "Article 366", as different.
5. For similar terms (e.g., "The Gazette of India" and "TheGAZETTE OF INDIA EXTRAORDINARY"), retain only one of the most representative terms.
6. Return only a Python list of the refined terms. Avoid adding any terms not present in the original list.
7. Do not provide any additional comments or explanations. Just return the refined list.
8. Say nothing else. For example, don't say: "Here are the keywords present in the document"

**Keywords:**
```{keywords}```

Ensure the final list is comprehensive, relevant, and free of duplicates. If there are any duplicates, you will be required to remove them before finalizing the list, otherwise die.
"""

CheckSummaryPrompt = """
You are an experienced legal analyst with a deep understanding of Indian law. Your task is to verify the accuracy and completeness of a summary in relation to various chunks of a document.
The summary will be enclosed in triple backticks (```). The chunks will be enclosed in the (#) tag.

For each document chunk and its corresponding summary:

1. **Understand Context and Key Points:**
   - Carefully read through the provided document chunk to understand its context, key points, and essential details.
2. **Accuracy Check:**
   - Compare the summary against the document chunk to ensure it accurately reflects all key points without omitting any important context.
3. **Maintain Precision and Enrich the Summary:**
   - Be precise in your revisions, ensuring that any added context is directly supported by the document chunk and relevant to the summary.
   - **Add details** from the document chunk to the summary if they are important and missing, ensuring the summary remains comprehensive.
4. **Revise the Summary if Necessary:**
   - If the summary misses any critical details or misrepresents the content of the document chunk, make the necessary corrections to ensure it is complete and accurate.
   - Ensure that the revised summary includes all essential information from the document chunk without shortening it.

**Output only the revised summary. If the summary is already accurate and complete, or if the document chunk is irrelevant, return the summary as is. Do not remove anything from the original summary.**

**Summary:**
```{summary}```
**Document chunk:**
#{text}#

**Revised Summary:**
"""

