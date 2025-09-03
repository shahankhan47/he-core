
from openai import OpenAI
import time
import os
import json
from db_operations import get_thread , update_thread_id

# Set OpenAI API Key from environment variable for security
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)


async def assistant_function(project_id :str,  assistant_action :str, content : str):

    ASSISTANT_ID = os.getenv(assistant_action)
    #check if the the thread exists
    try:
        thread_id = None
        thread_status = await get_thread( project_id = project_id, assistant_name = assistant_action)

        if thread_status != "thread does not exist":
            thread_id = thread_status
            messages_response = client.beta.threads.messages.create(
            
                role = "user",
                content= json.dumps(content),
                thread_id= thread_id
                
               )


        else :
            message_thread = client.beta.threads.create(
            messages=[
                {
                "role": "user",
                "content": json.dumps(content)
                },
               ]
               )
            thread_id = message_thread.id
            await update_thread_id(thread_id = thread_id, project_id = project_id, assistant_name= assistant_action)


        run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id= ASSISTANT_ID
    )

        # Wait for assistant's response
        
        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
            if run_status.status == "completed":
                break
            time.sleep(1)

    # Retrieve assistant response
        messages = client.beta.threads.messages.list(
            thread_id=thread_id
        )

        assistant_response = messages.data[0].content[0].text.value
        messages_response = client.beta.threads.messages.create(
            role = "assistant",
            content= assistant_response,
            thread_id= thread_id)
               
        


        return(assistant_response)

     
            
    except Exception as e:
        print(f"An error occurred while communicating with assistant: {e}")
                        


