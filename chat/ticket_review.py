import os
import chromadb
from chromadb.config import Settings
import chromadb.utils.embedding_functions as embedding_functions
from codebase import open_ai_token_counter
import json
from db_operations import get_project_details_by_id, get_summary_from_db
from openai import AsyncOpenAI
import asyncio

api_key = os.getenv('OPENAI_API_KEY')
open_ai_client = AsyncOpenAI(api_key=api_key)
chroma_host = os.getenv("CHROMA_HOST", "https://chroma.agreeablesky-9ecb3d35.eastus.azurecontainerapps.io")
chroma_port = os.getenv("CHROMA_PORT", 8000)


def split_string_into_chunks(text, chunk_size=200000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def query_chroma_db(query, project_id):
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
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
    return context


async def ticket_assistant(project_id, ticket_info):
    try:
        # 1. Retrieve project info and summary
        project_details = await get_project_details_by_id(project_id)
        if not project_details:
            return "Project not found in Harmony Engine."

        summary_content = await get_summary_from_db(project_details['owner_email'], project_id)
        chunks = split_string_into_chunks(summary_content)
        chunks = list(reversed(chunks))  # append last chunk closest to user turn

        # 2. Build messages: summary chunks, then ticket info
        messages = []
        for idx, chunk in enumerate(chunks):
            messages.append({
                "role": "user",
                "content": f"Project Summary (part {idx + 1}):\n{chunk}"
            })
        messages.append({
            "role": "user",
            "content": f"Ticket Information:\n{ticket_info}"
        })

        # 3. Initiate thread & run
        ASSISTANT_ID = os.getenv("ZENDESK_ASSISTANT")
        message_thread = await open_ai_client.beta.threads.create(
            messages=messages
        )
        thread_id = message_thread.id

        # (a) Count assistant messages at thread start (should be zero)
        initial_messages = await open_ai_client.beta.threads.messages.list(thread_id=thread_id)
        initial_assistant_count = sum(1 for msg in initial_messages.data if msg.role == "assistant")

        # (b) Start run
        run = await open_ai_client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=ASSISTANT_ID
        )

        # 4. Main tool run-poll loop: any number of rounds, any number of tool calls per round
        while True:
            run_status = await open_ai_client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            status = getattr(run_status, "status", None)

            # ---- Handle tool call(s) (ALL in batch) ----
            if (
                hasattr(run_status, "required_action") and
                run_status.required_action and
                run_status.required_action.type == "submit_tool_outputs"
            ):
                tool_outputs = []
                for tool_call in run_status.required_action.submit_tool_outputs.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    # RAG tool example (replace with your logic if needed)
                    if function_name == "rag_query_tool":
                        query = arguments.get("query") or arguments.get("rag_query", {}).get("Query")
                        db_result = query_chroma_db(query, project_id)
                        retrieved_db_result = await open_ai_token_counter.open_ai_truncator(text=db_result, max_tokens=160000, model = 'gpt-4o')
                        tool_output = f"\n\nRetrieved information:\n```\n{retrieved_db_result}\n```\n\n"
                    else:
                        tool_output = "Tool call not recognized or not implemented."

                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": tool_output
                    })
                await open_ai_client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
                await asyncio.sleep(2)
                # Loop again: more tool calls or done?
                continue

            # ---- Done? (No more tool calls requested) ----
            if status == "completed":
                # 5. Wait for NEW assistant message after completion (prevents premature return)
                for _ in range(120):  # max 30 sec (change as needed)
                    messages_res = await open_ai_client.beta.threads.messages.list(thread_id=thread_id)
                    assistant_messages = [msg for msg in messages_res.data if msg.role == "assistant"]
                    if len(assistant_messages) > initial_assistant_count:
                        last_assistant = sorted(assistant_messages, key=lambda m: m.created_at)[-1]
                        # cover both OpenAI SDK output structures
                        content = last_assistant.content[0]
                        if hasattr(content, "text"):
                            return content.text.value
                        elif isinstance(content, dict) and "text" in content:
                            return content["text"].get("value", "")
                        else:
                            return str(content)
                    await asyncio.sleep(1)
                return "No new assistant message generated after tool calls."
            # Still in progress
            await asyncio.sleep(2)

    except Exception as e:
        print(f"An error occurred while communicating with assistant: {e}")
        return f"Error: {str(e)}"