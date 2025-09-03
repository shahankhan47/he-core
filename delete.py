from db_operations import delete_project_data
import os
import chromadb
import asyncio


async def delete(project_id: str, email: str):
    try:
        await delete_collection(project_id)
        await delete_project_data(project_id=project_id, email=email)
        return "Deleted successfully"
    except Exception as e:
   
        # Return an error message
        return f"An error occurred while deleting project '{project_id}' for email '{email}': {e}"


async def delete_collection(project_id: str):
    # Configure ChromaDB
    chroma_host = os.getenv("CHROMA_HOST", "https://chroma.agreeablesky-9ecb3d35.eastus.azurecontainerapps.io")
    chroma_port = os.getenv("CHROMA_PORT", 8000)

    # Initialize Chroma client
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

    try:
        # Since ChromaDB's client methods are synchronous,
        # we can run them in a thread executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()

        # Check if the collection exists by running in an executor
        collections = await loop.run_in_executor(None, client.list_collections)
        collection_exists = any(
            collection.name == project_id for collection in collections
        )

        if collection_exists:
            # Delete the collection
            await loop.run_in_executor(None, client.delete_collection, project_id)
            print(f"Collection '{project_id}' deleted successfully.")
        else:
            print(f"Collection '{project_id}' does not exist.")

    except Exception as e:
        print(f"An error occurred while deleting the collection: {e}")