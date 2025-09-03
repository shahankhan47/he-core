import os
import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions
from codebase.open_ai_token_counter import open_ai_truncator
import json
from db_operations import get_project_details_by_id, get_summary_from_db
from openai import AsyncOpenAI
import asyncio
import re
from typing import List, Dict

api_key = os.getenv('OPENAI_API_KEY')
open_ai_client = AsyncOpenAI(api_key=api_key)
absolutepath = os.getenv("ABSOLUTE_PATH")

"""def split_string_into_chunks(text, chunk_size=200000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
"""

"""def query_chroma_db(query, project_id):
    client = chromadb.PersistentClient(path=absolutepath)
    collection_name = project_id
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv('OPENAI_API_KEY'),
        model_name="text-embedding-3-large"
    )
    collection = client.get_collection(name=collection_name, embedding_function=openai_ef)

    results = collection.query(
        query_texts=query,
        n_results=4
    )

    context = ""
    for result in results['documents'][0]:
        context += f"results: {result}\n"
    return context"""


async def pr_review(project_id, git_diff, deep_review):
    try:
        # 1. Retrieve project info and summary
        project_details = await get_project_details_by_id(project_id)
        if not project_details:
            return "Project not found in Harmony Engine."

        summary_content = await get_summary_from_db(project_details['owner_email'], project_id)

        # 2. Split the git diff by file
 
        async def split_diff_by_file(diff_text: str):
            # Use lookahead to split but keep splitters
            # Works for both multi-line and single-line diffs if `diff --git` always starts a new file
            files = []
            parts = re.split(r"(?=diff --git )", diff_text)
            for part in parts:
                if not part.strip():
                    continue
                m = re.match(r'diff --git a/(\S+) b/(\S+)', part)
                fname = m.group(2) if m else "unknown"
                files.append({"filename": fname, "diff": part})
            return files
        file_diffs = await split_diff_by_file(git_diff)
        summaries = []

        # 3. For each file's diff, call OpenAI to get a summary
        for file in file_diffs:
            
            try:
                diiff_truncated =  await open_ai_truncator(text =  file['diff'], model= "gpt-4o" , max_tokens= 80000)
                response = await open_ai_client.chat.completions.create(
                    model="gpt-4o", # or your preferred model
                    messages=[
                        {"role": "system", "content": "You will be provied with the git diff of a file, summarize the change in a succint and simple."},
                        {"role": "user", "content":diiff_truncated }
                    ]
                )
                summary_text = response.choices[0].message.content.strip()
            except Exception as err:
                summary_text = f"OpenAI error: {str(err)}"
            summaries.append(f"### {file['filename']}\n{summary_text}")

        # 4. Collated summary
        final_summary = "\n\n".join(summaries)
        return final_summary

    except Exception as e:
        print(f"An error occurred while communicating with assistant: {e}")
        return f"Error: {str(e)}"