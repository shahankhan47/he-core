import os
import json
import re
import asyncio
from openai import AsyncOpenAI
import chromadb
from chromadb.utils import embedding_functions
import chat.assistant as assistant
from db_operations import store_conversation_in_db, get_conversation_history_from_db
from codebase.open_ai_token_counter import open_ai_truncator
import time

# Load environment variables
api_key = os.getenv('OPENAI_API_KEY')
chroma_host = os.getenv("CHROMA_HOST", "https://chroma.agreeablesky-9ecb3d35.eastus.azurecontainerapps.io")
chroma_port = os.getenv("CHROMA_PORT", 8000)
open_ai_client = AsyncOpenAI(api_key=api_key)

# Initialize OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name="text-embedding-3-large"
)

query_cache = {}

# Summarize early messages
async def summarize_early_exchanges(messages: list[dict], num_exchanges: int = 4) -> str:
    if len(messages) <= num_exchanges:
        return ""

    early_messages = messages[:num_exchanges * 2]
    summary_prompt = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in early_messages])
    input_text = f"Summarize the following conversation concisely:\n\n{summary_prompt}"

    response = await open_ai_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "system", "content": "You summarize conversations concisely."},
                  {"role": "user", "content": f"Provide a brief summary. Here is the conversation: {input_text}"}],
        max_tokens=500,
        temperature=0.2
    )
    return response.choices[0].message.content

def query_chroma_db(collection, query):
    if query in query_cache:
        return query_cache[query]
    results = collection.query(query_texts=query, n_results=8)
    if not results['documents'][0]:
        output = "No relevant codebase content found for the generated query."
    else:
        output = "\n".join([f"results: {doc}" for doc in results['documents'][0]])
    query_cache[query] = output
    return output

async def create_checklist(project_id, tool_name, tool_call):
    await assistant.assistant_function(
        project_id=project_id,
        assistant_action=tool_name,
        content=tool_call.function.arguments
    )

async def store_in_db(email_id, project_id, user_question, raw_response_full, checklist_created=False, checklist_title="", checklistAssistant=False):
    print(f"Checklist created: {checklist_created}, Checklist Assistant called: {checklistAssistant}, checklist title: {checklist_title}")
    if not checklistAssistant:
        await store_conversation_in_db(email_id, project_id, "user", user_question)
        await store_conversation_in_db(email_id, project_id, "assistant", raw_response_full)
    elif checklist_created and checklistAssistant:
        await store_conversation_in_db(email_id, project_id, "user", f"User requested checklist creation")
        await store_conversation_in_db(email_id, project_id, "assistant", f"Checklist created: {checklist_title}. The checklist will appear in the 'Task Checklist' section after some time. If not, please refresh the page to view it.")
    else:
        print("Checklist Assistant was called but no checklist was created.")
        await store_conversation_in_db(email_id, project_id, "user", f"User requested checklist creation")
        await store_conversation_in_db(email_id, project_id, "assistant", f"Checklist Assistant was called but no checklist was created. Please try again later.")

async def chat(email_id: str, project_id: str, summary: str, user_question: str, checklistAssistant: bool, uploaded_files: list, on_stream: callable) -> str:
    try:
        t0 = time.monotonic()
        client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        collection = client.get_collection(name=project_id, embedding_function=openai_ef)

        # Parallelize all async calls
        conversation_task = get_conversation_history_from_db(email_id, project_id)
        summary_task = open_ai_truncator(text=summary, max_tokens=10000, model="gpt-4.1-mini")
        conversation_messages, summary = await asyncio.gather(conversation_task, summary_task)

        system_message = [
            {"role": "system", "content": "You are a senior software architect expert in code analysis."},
            {"role": "system", "content": f"Codebase summary: {summary}"},
            {"role": "system", "content": "Checklist = call CHECKLIST_ASSISTANT. Missing code = call Querycodebase."}
        ]
        t1 = time.monotonic()

        api_messages = []
        if conversation_messages:
            early_summary_task = summarize_early_exchanges(conversation_messages[:-2])
            last_msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": msg["content"]}
                         for i, msg in enumerate(conversation_messages[-2:])]
            api_messages.extend(last_msgs)
            early_summary = await early_summary_task
            if early_summary:
                system_message.append({"role": "system", "content": f"Earlier summary: {early_summary}"})
        api_messages.append({"role": "user", "content": user_question})

        force_checklist = bool(re.search(r'\b(create|build|generate)\b.*\bchecklist\b', user_question, re.IGNORECASE))

        # Step 1: Generate a ChromaDB search query
        query_prompt = system_message + api_messages + [
            {"role": "system", "content": "Generate a vector search query to retrieve code snippets or documents."}
        ]
        query_response = await open_ai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=query_prompt,
            max_tokens=500,
            temperature=0.3
        )
        t2 = time.monotonic()
        rag_query = query_response.choices[0].message.content

        # Step 2: Query DB and format result
        db_result = query_chroma_db(collection, rag_query)
        retrieved_context = await open_ai_truncator(text=db_result, max_tokens=20000, model="gpt-4.1-mini")

        t3 = time.monotonic()
        system_message.extend([
            {"role": "system", "content": f"Search query: {rag_query}"},
            {"role": "system", "content": f"Retrieved codebase info: {retrieved_context}"},
            {"role": "system", "content": f"Original User question: {user_question}"}
        ])

        if uploaded_files:
            system_message.extend([
                {"role": "system", "content": f"""The user has also uploaded one or more files. 
                Here are the names and summaries of the files (Each object contains a fileName and fileSummary):
                {uploaded_files}
                """},
            ])

        final_prompt = system_message + api_messages + [
            {"role": "system", "content": """
                Analyze all context and provide a precise, implementation-ready answer to the question.
                If explaining multiple items (components, files, functions, etc.):
                - List each explicitly in a structured format (numbered list or table)
                - Explain purpose, functionality, dependencies, and key implementation details of each
                - Don’t skip or merge items, even minor ones
                - Include examples, configs, code-snippets, relationships, and usage considerations where relevant
                Make the answer exhaustive and developer-friendly.
            """},
        ]

        if "No relevant codebase content found" in db_result:
            final_prompt.append({"role": "user", "content": f"Call Querycodebase with query: {rag_query}."})

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "CHECKLIST_ASSISTANT",
                    "description": "Create a technical checklist with specific tools, versions, commands.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Title": {"type": "string", "description": "Checklist title."},
                            "checklist": {"type": "string", "description": "Detailed checklist content."}
                        },
                        "required": ["Title", "checklist"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "Querycodebase",
                    "description": "Run a code-level search using a specific query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Query": {"type": "string", "description": "The query to run."}
                        },
                        "required": ["Query"]
                    }
                }
            }
        ]

        t4 = time.monotonic()
        # Step 3: Tool interaction loop
        checklist_created = False
        checklist_title = ""
        while True:
            model_response = await open_ai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=final_prompt,
                max_tokens=2000,
                tool_choice="auto",
                tools=tools
            )
            finish_reason = model_response.choices[0].finish_reason

            if finish_reason != "tool_calls" and force_checklist:
                force_checklist = False
                final_prompt.append({
                    "role": "system",
                    "content": "Checklist requested. Please call the CHECKLIST_ASSISTANT tool now."
                })

            elif finish_reason == "tool_calls":
                for tool_call in model_response.choices[0].message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    if tool_name == "Querycodebase":
                        query = tool_args.get("Query", "")
                        tool_result = query_chroma_db(collection, query)
                        formatted_result = await open_ai_truncator(tool_result, max_tokens=20000, model="gpt-4.1-mini")
                        final_prompt.append({"role": "user", "content": formatted_result})

                    elif tool_name == "CHECKLIST_ASSISTANT":
                        checklist_title = tool_args.get("Title", "unspecified checklist")
                        force_checklist = False
                        final_prompt.append({
                            "role": "system",
                            "content": f"""IMPORTANT: The checklist titled '{checklist_title}' is being created in the background. 
                            Please inform the user to REFRESH THE PAGE to load the checklist. 
                            DO NOT repeat this process AND DO NOT call the CHECKLIST_ASSISTANT tool again.
                            Make the final response only inform the user that the checklist is being created and to refresh the page later."""
                        })
                        checklist_created = True
            else:
                break

        t5 = time.monotonic()
        print(f"Initial setup time: {t1 - t0:.2f} sec")
        print(f"Early summary and query generation time: {t2 - t1:.2f} sec")
        print(f"ChromaDB search time: {t3 - t2:.2f} sec")
        print(f"Variables time: {t4 - t3:.2f} sec")
        print(f"Tools run time: {t5 - t4:.2f} sec")
        print(f"Total time: {t5 - t0:.2f} sec")

        raw_response_full = ""
        if not checklistAssistant:
            final_response_stream = await open_ai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=final_prompt,
                max_tokens=10000,
                stream=True
            )

            async for chunk in final_response_stream:
                delta = chunk.choices[0].delta
                content = delta.content
                if content:
                    raw_response_full += content
                    await on_stream(content)

        else:
            res = f"Checklist created: {checklist_title}. The checklist will appear in the 'Task Checklist' section after some time. If not, please refresh the page to view it."
            for content in res:
                await on_stream(content)

        asyncio.create_task(store_in_db(email_id, project_id, user_question, raw_response_full, checklist_created, checklist_title, checklistAssistant))
        if checklist_created:
            asyncio.create_task(create_checklist(project_id, tool_name, tool_call))

    except Exception as e:
        await on_stream(f"[ERROR] {str(e)}")


async def analyze_image(data_uri: str):
    try:
        response = await open_ai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_uri
                            }
                        },
                        {
                            "type": "text",
                            "text": "Analyze the image"
                        }
                    ]
                }
            ],
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception("Error processing image: ", e)


async def analyze_csv_or_excel(excelText: str):
    try:
        response = await open_ai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f"Analyze the following spreadsheet data and provide insights:\n\n{excelText}"
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception("Error processing speadsheet: ", e)