#$~ Harmony Engine V3 ~$############################################################################################################################
#$~ Written by thedataguy ~$#
"""
There are redundant function definitions throughout the api routes, its done to keep all routes independent.
The idea is that at one point we can turn the api into microservices architecture without introducing complex interdependencies.
Along the same lines, Im trying out a architecture where the code block become self contained since its mostly by AI 
"""
import base64
import fnmatch
import os
import zipfile
import tempfile
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Form, Query, Body, Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from pydantic import BaseModel 
import aiofiles  
import shutil
import uuid
import json
from fastapi.responses import JSONResponse
import tiktoken
from typing import List
import asyncio
from dotenv import load_dotenv
import sys
#Local Dependencies 
from db_operations import store_summary_in_db, get_summary_from_db, update_status_in_db, get_conversation_history_from_db, get_executive_summary_from_db, get_project_diagrams_from_db, create_pin_in_db, delete_pin_from_db, get_pins_from_db, update_project_in_db
from db_operations import get_user_projects as fetch_user_projects, get_project_details_by_id,  add_user_to_project,  delete_user_from_project, get_project_details_by_id
from codebase.anthropic_token_counter import anthropic_truncator
from chat.ticket_review import ticket_assistant
from chat.pr_review import pr_review
from chat.pr_review import pr_review
from codebase.summarizer import email_analysis_failed, email_summary, summarize_file, analyze_summary_with_anthropic, update_vectors,  generate_questions_openai, generate_responses, generate_executive_summary, generate_project_diagrams
import chat.chat_pro as chat_pro
import chat.chat_lite as chat_lite
import chat.mermaid  as mermaid
from delete import delete
from typing import Dict
from typing import Optional
import io
from codebase.pdf import PDF, clean_markdown_codeblock
import chat.assistant as assistant
from fastapi import Security, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.security.api_key import APIKeyHeader
from starlette.responses import JSONResponse
import httpx
import re
from fastapi import FastAPI


#$~ Configs ~$############################################################################################################################

load_dotenv()


token_limit = 170000 # based on open ai

"""Update to support codebases after testing"""
TEXT_FILE_EXTENSIONS = { '.dart', '.md', '.markdown', '.js', '.ts', '.py', '.cs', '.cpp', '.c', '.h', 
                        '.hpp', '.java', '.kt', '.kts', '.rb', '.php', '.html', '.css', 
                        '.scss', '.less', '.xml', '.json', '.yml', '.yaml', '.toml', 
                        '.ini', '.cfg', '.txt', '.sh', '.bat', '.ps1', '.rs', '.go', 
                        '.swift', '.m', '.mm', '.pl', '.pm', '.r', '.jl', '.scala', 
                        '.lua', '.sql', '.erl', '.hrl', '.ex', '.exs', '.dart', '.groovy', 
                        '.f90', '.f95', '.f03', '.f08', '.vb', '.vbs', '.asm', '.s', 
                        '.lhs', '.hs', '.tsx', '.jsx', '.vue', '.ada', '.adb', '.ads', 
                        '.d', '.e', '.factor', '.forth', '.ftl', '.ml', '.mli', '.mlp', 
                        '.mly', '.pp', '.pwn', '.pug', '.razor', '.cshtml', '.tpl', '.agc', '.ipynb'}


DEFAULT_IGNORE_PATTERNS = [
    # Directories
    'node_modules', 'venv', '.git', '__pycache__',  'build',  'dist',
    # File types
    '*.pyc',  '*.pyo',  '*.so',   '*.o',   '*.obj',   '*.exe',   '*.dll',   '*.bin',   '*.log',   '*.tmp',   '*.bak',
    #Flutter
    '.dart_tool', '.flutter-plugins', '.flutter-plugins-dependencies', '.plugin_symlinks', 'build', '*.g.dart',  '*.freezed.dart',  '*.mocks.dart', '*.config.dart',  'ios/Pods', 'android/.gradle'
    
]
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )

combined_app = FastAPI()
combined_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify allowed frontend origins
    allow_methods=["*"],
    allow_headers=["*"],
)
app = FastAPI(dependencies=[Depends(get_api_key)])
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=["*"],  
    allow_headers=["*"],  
)

"""All Classes need to be revised, this bit is convoluted"""
class AnalyzeSummaryRequest(BaseModel):
    summary: str
    user_question: str

class Project(BaseModel):
    project_id: str
    created_at: str
    status: str
    project_name :str
    role: Optional[str] 
    project_description: Optional[str]
    file_source: Optional[str]
    commit_id: Optional[str]  


class ProjectsResponse(BaseModel):
    projects: List[Project]

class Message(BaseModel):
    role: str
    content: str

class ConversationHistoryResponse(BaseModel):
    history: List[Message]
    is_new_chat: bool

class TicketReviewRequest(BaseModel):
    project_id: str
    ticket_content: str
    ticket_id: str          # <-- New field
    callback_url: str

class PRReviewRequest(BaseModel):
    project_id: str
    pr_id: str
    deep_review: str
    git_diff: str



#$~ API 1 ~$############################################################################################################################
#$~ Description ~$#
"""
This route takes in a zip and emails the files after processing by open ai 4.0 mini model.
Open AI is used specificaly to spread the API request load and avoid rate throttling.
Open AI is the primary code to Summary LLM.
"""

#$~ Task List ~$#
"""
1) Needs error handling on no files found


"""

@app.post("/addcodebase", description="This api takes a codebase or any collection of files as a zip, along with an email. The email will be used to identify the user. Once the intermediate descriptions are generated, a request ID will be provided. This request ID and email id uniquely identify the codebase being processed. A Email notification will be sent to the user")
async def add_codebase( file: UploadFile, project_id: str, background_tasks: BackgroundTasks, file_source: str, commit_id: str ):

    
    # Check if the codebase uploaded is actually a zip
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a zip file.")
    
    # Temporary directory created in the file system and deleted later
    temp_dir = tempfile.mkdtemp()
    project_details = await get_project_details_by_id(project_id)

    # Create initial entry in database asynchronously
    await update_status_in_db(emails = project_details["emails"], project_id=project_id, project_description = project_details["project_description"],   project_name = project_details["project_name"],  status="Initiated", summary = None, executive_summary= None, project_diagrams = None, file_source = file_source, commit_id = commit_id)
    try:
        zip_path = os.path.join(temp_dir, file.filename)

        # Save the uploaded zip file
        async with aiofiles.open(zip_path, 'wb') as f:  
            contents = await file.read()  
            await f.write(contents)  

        # Extract the contents of the zip file
        extract_dir = os.path.join(temp_dir, 'extracted_files')
        os.makedirs(extract_dir)

        ### Helper for Zip Extraction
        def extract_zip_file(zip_path: str, extract_to: str):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
             zip_ref.extractall(extract_to)

        loop = asyncio.get_event_loop()  # Acquire the current event loop
        await loop.run_in_executor(None, extract_zip_file, zip_path, extract_dir) 


        # Process files in the background
        background_tasks.add_task(process_and_post_summary, extract_dir, project_details= project_details, project_id= project_id, file_source = file_source, commit_id = commit_id)
        
        return {"detail": "File upload successful, generating description in background.",
                "project_id": project_id,
                "project_name" :project_details["project_name"]}
    
    except Exception as e:
        shutil.rmtree(temp_dir)
        await update_status_in_db(emails = project_details["emails"], project_id=project_id, project_description = project_details["project_description"],   project_name = project_details["project_name"] , status = f"Stage 1 Error during file upload: {str(e)}" , summary = None, executive_summary= None, project_diagrams = None, file_source = file_source, commit_id = commit_id)
        raise HTTPException(status_code=500, detail="Internal server error")
    

# Iterate through each file in zip and call Open AI to generate the summary 

#`````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
def extract_first_json(text):
    """Extracts the first JSON object found in a string (non-greedy)."""
    match = re.search(r'\{(?:[^{}]|(?R))*\}', text)
    if match:
        return match.group(0)
    return None

def safe_json_loads(possible_json):
    """Try to load JSON, and extract if embedded or duplicated."""
    try:
        return json.loads(possible_json)
    except Exception:
        extracted = extract_first_json(possible_json)
        if extracted:
            try:
                return json.loads(extracted)
            except Exception:
                pass
        # As a last resort:
        print("Failed to parse summary as JSON:\n", possible_json)
        raise



def should_ignore_path(path, ignore_patterns=DEFAULT_IGNORE_PATTERNS):
    path_parts = path.split(os.sep)
    for pattern in ignore_patterns:
        if any(fnmatch.fnmatch(part, pattern) for part in path_parts):
            return True
    return False


async def process_file(file_path, extract_dir):
    relative_path = os.path.relpath(file_path, extract_dir)

    if should_ignore_path(relative_path):
        return None, None  # Skip ignored files

    if os.path.splitext(file_path)[1].lower() in TEXT_FILE_EXTENSIONS:
        file_path_1, content, summary = await summarize_file(file_path)
        print(f'Summary succeeded for {file_path}')
        summary_dict = safe_json_loads(summary)

        qualitative_score = summary_dict.get('qualitative_score', 0)
        summary = summary_dict.get('summary', '')

        if summary:
            file_name = os.path.basename(file_path)
            context_summary = {
                'score': qualitative_score,
                'text': f"file name is {file_name}, \nfile path is: {file_path_1}, \nfilesummary is: {summary}"
            }
            full_summary = [file_name, file_path, content, summary]
            return context_summary, full_summary

    return None, None

async def process_and_post_summary(extract_dir: str, project_id: str, project_details: str, file_source: str, commit_id: str):
    try:
        # Status update in db
        await update_status_in_db(emails=project_details["emails"], project_id=project_id, 
                                project_description=project_details["project_description"],
                                project_name=project_details["project_name"],
                                status="Summary being generated", summary="", executive_summary="", 
                                project_diagrams="", file_source=file_source, commit_id=commit_id)

        # Gather all file paths first
        all_file_paths = []
        for root, dirs, files in os.walk(extract_dir):
            dirs[:] = [d for d in dirs if not should_ignore_path(os.path.join(root, d), DEFAULT_IGNORE_PATTERNS)]
            
            for name in files:
                file_path = os.path.join(root, name)
                if os.path.splitext(file_path)[1].lower() in TEXT_FILE_EXTENSIONS and not should_ignore_path(file_path):
                    all_file_paths.append(file_path)
        
        # Process all files in parallel
        async def process_files_in_parallel(file_paths, batch_size=10):
            context_summaries = []
            full_summaries = []
            
            # Process files in batches to avoid overwhelming the system
            for i in range(0, len(file_paths), batch_size):
                batch = file_paths[i:i+batch_size]
                tasks = [process_file(file_path, extract_dir) for file_path in batch]
                results = await asyncio.gather(*tasks)
                
                for context_summary, full_summary in results:
                    if context_summary and full_summary:
                        context_summaries.append(context_summary)
                        full_summaries.append(full_summary)
            
            # Sort context summaries by score in descending order
            context_summaries.sort(key=lambda x: x['score'], reverse=True)
            combined_summary = '\n\n\n'.join([item['text'] for item in context_summaries])
            
            return combined_summary, full_summaries
        
        context_summaries, full_summaries = await process_files_in_parallel(all_file_paths)

        ###
        if context_summaries:
            combined_summary = anthropic_truncator(text =  context_summaries)


            await update_vectors(project_id= project_id, full_summaries = full_summaries, action = 'create')

            executive_summary = await generate_executive_summary(combined_summary)

            diagrams = await generate_project_diagrams(project_id =project_id,summary= combined_summary )

            await store_summary_in_db(emails= project_details["emails"], project_id= project_id, summary= combined_summary, status="Stage 1 Completed", executive_summary = executive_summary, project_diagrams = diagrams)

            email_summary(combined_summary, project_details["emails"], project_id, project_details["project_name"])
            print("email sent")
        else:
            await update_status_in_db(emails = project_details["emails"], project_id=project_id, project_description = project_details["project_description"],   project_name = project_details["project_name"], status=  "Stage 1 No summaries generated, Contact: support@harmonyengine.ai ", summary = None, executive_summary= None, project_diagrams = None, file_source = file_source, commit_id = commit_id)
            print("no files were found")
            pass
    except Exception as e:
        error_msg = f"Error in process_and_post_summary: {str(e)}"
        print(error_msg)
        await update_status_in_db(emails = project_details["emails"], project_id=project_id, project_description = project_details["project_description"],   project_name = project_details["project_name"], status = error_msg,  summary = None, executive_summary = None,  project_diagrams = None,  file_source = file_source, commit_id = commit_id)
        await email_analysis_failed(project_details["emails"], project_details["project_name"])


    finally:
            shutil.rmtree(extract_dir)

#$~ API 2 ~$############################################################################################################################
#$~ Description ~$#
"""
This route fetches the summary based on request id and email.
based on that generates questions, then chats with the codebase summary to pull outmore details
Anthropic for chat, Open AI to convert the response to JSON.
Depricated
"""

#$~ Task List ~$#
"""
- Depriicated 

"""

@app.post("/generate-summary", description="   WARNINING DEPRICATED: Generates a comprehensive report of the codebase.")

async def analyze_summary(background_tasks: BackgroundTasks, email: str = Form(...), project_id: str = Form(...)):
   
    try:
        # Fetch summary from PostgreSQL
        summary_content = await get_summary_from_db(email, project_id)
        if not summary_content:
            #update_status_in_db(email, project_id, "Stage 2 Error: Summary not found")
            raise HTTPException(status_code=404, detail="Summary not found")

        enc = tiktoken.encoding_for_model("gpt-4")  # Modify as needed for Anthropic
        tokens = enc.encode(summary_content)
        token_count = len(tokens)
        if token_count > token_limit:
            truncated_tokens = tokens[:token_limit]
            summary_content = enc.decode(truncated_tokens)

        #update_status_in_db(email, project_id, "STAGE 2 Summary retrieved successfully")

    except Exception as e:
        #update_status_in_db(email, project_id, f"Error: STAGE 2 Failed to retrieve summary - {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to retrieve summary: {str(e)}")

    background_tasks.add_task(process_summary, email, summary_content, project_id)
    
    return JSONResponse(content={"detail": "Summary retrieved successfully, processing in background. You will receive an email shortly"})

async def process_summary(email: str, summary_content: str, project_id: str):

    # use the summary to set up a base set of questions from anthropic, this is then fed to open ai to get structured questions. 
    #either use anthropic fully - needs the response to be in json or use open ai fully - right now this is hack code
    question, executive_summary = analyze_summary_with_anthropic(summary_content)   ##############verify if this step is needed
   
    try:
       
        #update_status_in_db(email, project_id, "STAGE 2 Processing with OpenAI assistant")

        final_question =  await generate_questions_openai(question)
        print("generated : questions /n " )
        await generate_responses(final_question , email, summary_content, project_id, executive_summary )
    
    except Exception as e:
        #update_status_in_db(email, project_id, f"STAGE 2 Error in process_summary: {str(e)}")
        print(f"Error in process_summary: {e}")



#$~ API 3 ~$############################################################################################################################
#$~ Description ~$#
"""
This route fetches the summary from DB and uses Anthropic Prompt caching to let users chat with the codebase.
This is currently depricated >> Retaining the code so that I can turn this route into a Lite-chat for free users.
Anthropic for chat.
Message History - Last 5 turns of the conversation.
Depricated
"""

#$~ Task List ~$#
"""
1) Model encoding - change it to anthropic tokenizer.
Depricated
"""

@app.post("/chat-lite", description= "  WARNINING DEPRICATED: This API is used to chat with the master summary. Provides high level summaries only. The Application remembers the last 5 conversations that the user had with the code.")
async def chat_lite_version(user_question: str = Form(...), project_id: str = Form(...) ,email: str = Form(...) ):
    try:
        # Fetch summary from PostgreSQL
        summary_content = await get_summary_from_db(email, project_id)
        if not summary_content:
            raise HTTPException(status_code=404, detail="Summary not found")

        enc = tiktoken.encoding_for_model("gpt-4o")   #needs to be modified to anthropic
        tokens = enc.encode(summary_content)
        token_count = len(tokens)
        if token_count > token_limit:
            # Truncate the summary to match the token limit
            truncated_tokens = tokens[:token_limit]
            summary_content= enc.decode(truncated_tokens)  # Decode tokens back to string

    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to read summary file")
    

    response = await  chat_lite.codebase_qa_with_anthropic(email, project_id, summary_content, user_question)
    return {"result": response}




#$~ API 4 ~$############################################################################################################################
#$~ Description ~$#
"""
This route fetches all projects associated with a user

"""

#$~ Task List ~$#
"""
1) Project Name parameter.
"""

@app.get("/projects", description="This API is used to retrieve all projects for a given user. Provide the user's email to get their project list.")
async def get_user_projects(
    email: str = Query(..., description="The email address of the user")
):
    try:
        # Use the renamed function here
        projects_json = await fetch_user_projects(email)
        projects_data = json.loads(projects_json)
        return ProjectsResponse(**projects_data)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON data returned from database")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


#$~ API 5 ~$############################################################################################################################
#$~ Description ~$#
"""
This API retrieves the conversation history for a specific user and request. 
It returns the last 5 conversations and indicates if it's a new chat.

"""

#$~ Task List ~$#


@app.get("/conversation-history", description="This API retrieves the conversation history for a specific user and request. It returns the last 5 conversations and indicates if it's a new chat.")
async def get_conversation_history(
    email: str = Query(..., description="The email address of the user"),
    project_id: str = Query(..., description="The unique identifier for the request")
):
    try:
        history = await get_conversation_history_from_db(email, project_id)
        
        is_new_chat = len(history) == 0
        
        return ConversationHistoryResponse(
            history=[Message(**msg) for msg in history],
            is_new_chat=is_new_chat
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

    
#$~ API 6 ~$############################################################################################################################
#$~ Description ~$#

"""
This websocket fetches the summary from DB and uses OpenAI Prompt caching to let users chat with the codebase.
The api feteches an existing chroma db collection under the project Id / Project Id and uses tool rag for generating answers
OpenAI is used for chat. The response is returned in stream.
Message History - Last 5 turns of the conversation.
"""

@combined_app.websocket("/chat-pro")
async def chat_pro_version (websocket: WebSocket):
    await websocket.accept()
    try:
        # 1. Receive JSON payload from frontend
        data = await websocket.receive_json()
        user_question = data.get("user_question", "")
        project_id = data.get("project_id")
        email = data.get("email")
        checklistAssistant = data.get("checklistAssistant", False)
        uploaded_files_raw = data.get("uploaded_files")

        # Always convert to a Python list
        if not uploaded_files_raw:
            uploaded_files = []
        elif isinstance(uploaded_files_raw, str):
            try:
                parsed = json.loads(uploaded_files_raw)
                uploaded_files = parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                uploaded_files = []
        elif isinstance(uploaded_files_raw, list):
            uploaded_files = uploaded_files_raw
        else:
            uploaded_files = []

        async def on_stream(content_chunk: str):
            await websocket.send_text(content_chunk)

        # Fetch summary from PostgreSQL
        summary_content = await get_summary_from_db(email, project_id)
        if not summary_content:
            raise HTTPException(status_code=404, detail="Summary not found")
        
        enc = tiktoken.encoding_for_model("gpt-4")  # Modify as needed for Anthropic
        tokens = enc.encode(summary_content)
        token_count = len(tokens)
        if token_count > token_limit:
            truncated_tokens = tokens[:token_limit]
            summary_content = enc.decode(truncated_tokens)


    except Exception as e:
        await websocket.send_text(f"[ERROR] Failed to read summary: {str(e)}")

    try:
        await chat_pro.chat(email, project_id, summary_content, user_question, checklistAssistant, uploaded_files, on_stream)

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        await websocket.send_text(f"[ERROR] {str(e)}")
    finally:
        await websocket.close()

#$~ API 7 ~$############################################################################################################################
#$~ Description ~$#

"""
Depricated
This route fetches generates the mermaid diagram - takes up to 1 minute to return a response.
"""

#$~ Task List ~$#
#1 Currently place holder


@app.post("/generate-mermaid", description=" WARNING: Depricated. This API uses Caude and. Use the same email and request id as Stage 2. T")
async def generate_mermaid_diagrams (user_question: str = Form(...), project_id: str = Form(...), email: str = Form(...)):
    try:
        # Fetch summary from PostgreSQL
        summary_content = await get_summary_from_db(email, project_id)
        if not summary_content:
            raise HTTPException(status_code=404, detail="Summary not found")

        # generate summary
        result = await  mermaid.generate_diagrams(user_question = user_question,  project_id = project_id, email_id = email, summary = summary_content)
        
        return {"result": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in mermaid diagram generation: {str(e)}")
        
        


#$~ API 8 ~$############################################################################################################################
#$~ Description ~$#

"""
This route deletes a project.
"""

#$~ Task List ~$#
#1 Currently place holder


@app.post("/delete-project", description="This API will delete the Project summary, associated converstations and the vector collection")
async def delete_project (project_id: str = Form(...), email: str = Form(...)):
    try:
        result = await delete(  project_id = project_id, email = email)
        return {"result": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in deletion: {str(e)}")
        
        
#$~ API 9 ~$############################################################################################################################
#$~ Description ~$#
"""
This API retrieves Executive Summary of the project, this replaces the previous get summary api - API 2 

"""

#$~ Task List ~$#


@app.get("/executive-summary", description="This API retrieves the Executive Summary of the project and user.")
async def get_executive_summary(
    email: str = Query(..., description="The email address of the user"),
    project_id: str = Query(..., description="The unique identifier for the project")
):
    try:
        executive_summary = await get_executive_summary_from_db(email, project_id)
        
        
        
        return  executive_summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

#$~ API 10 ~$############################################################################################################################
#$~ Description ~$#
"""
This API retrieves project diagrams this replaces the previous generate mermaid api - API 7 

"""

#$~ Task List ~$#


@app.get("/project_diagrams", description="This API retrieves the project's mermaid diagrams.")
async def get_project_diagram(
    email: str = Query(..., description="The email address of the user"),
    project_id: str = Query(..., description="The unique identifier for the project")
):
    try:
        project_diagrams = await get_project_diagrams_from_db(email, project_id)
        
        
        
        return  project_diagrams
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




#$~ API 11 ~$############################################################################################################################
#$~ Description ~$#
"""
This API creates a pin 

"""

#$~ Task List ~$#


@app.post("/create_pin", description="This API creates a user pin.")
async def create_pin(
    email: str = Query(..., description="The email address of the user"),
    project_id: str = Query(..., description="The unique identifier for the project"),
    topic_name: str = Body(..., description="The topic for the pin"),
    pin_content: str = Body(..., description="The Pin content")
):
    try:
        await create_pin_in_db(email, project_id, topic_name, pin_content)
        return {"detail": "Pin Creation successful."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#$~ API 12 ~$############################################################################################################################
#$~ Description ~$#
"""
This API deletes a pin
"""

@app.delete("/delete_pin", description="This API deletes a user pin.")
async def delete_pin(
    email: str = Query(..., description="The email address of the user"),
    pin_id: int = Query(..., description="The ID of the pin to be deleted")
):
    try:
        await delete_pin_from_db(email, pin_id)
        return {"detail": "Pin deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


### Fetch Pins by Project API


#$~ API 13 ~$############################################################################################################################
#$~ Description ~$#
"""
This API fetches all pins for a specific project.
"""

@app.get("/fetch_pins_by_project", description="This API fetches all pins for a specific project.")
async def fetch_pins_by_project_route(
    email: str = Query(..., description="The email address of the user"),
    project_id: str = Query(..., description="The unique identifier for the project")
):
    try:
        pins = await get_pins_from_db(email, project_id)
        if pins is None or len(pins) == 0:
            raise HTTPException(status_code=404, detail="No pins found for this project.")
        return {"pins": pins}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


#$~ API 14 ~$############################################################################################################################
#$~ Description ~$#
"""
Initiates a blank project with 

"""

@app.post("/initialize_project", description="This api takes a project name and associated emails , project name , project description,  ")
async def initialize_project(owner_email: str,  project_name: str, project_description: str, emails: Dict[str, str], background_tasks: BackgroundTasks):
    try:
    # This will serve as the project id for the call
        project_id = str(uuid.uuid4())

        
        await update_project_in_db(project_id, owner_email, project_name, project_description, emails)
        #formatted_emails = json.dumps(emails)
        formatted_emails = [{"email": email, "role": role} for email, role in emails.items()]
        formatted_emails += [{"email": owner_email, "role": "owner"}]
        await update_status_in_db(emails = json.dumps(formatted_emails), project_id=project_id, project_description = project_description,   project_name = project_name,  status="Awating codebase", summary = None, executive_summary= None, project_diagrams = None, file_source = None, commit_id = None)

        return {"detail": "project creation successful.",
                    "project_id": project_id,
        }
    
                    
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")




 #$~ API 15 ~$############################################################################################################################
#$~ Description ~$#
"""
Adds users to an existing project 

"""

@app.post("/add_collabrator", description="This api takes an existing project id and adds a user with a role,  ")
async def add_user( project_id: str,  owner_email: str,  emails: Dict[str, str]):
 
    try:

        formatted_emails = [{"email": email, "role": role} for email, role in emails.items()]
        await add_user_to_project( project_id= project_id, users = json.dumps(formatted_emails), owner_email= owner_email )

        return {"detail": "user / users added.",
                    "project_id": project_id,
        }
    
                    
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")   
    

 #$~ API 16 ~$############################################################################################################################
#$~ Description ~$#
"""
Adds users to an existing project 

"""

@app.post("/delete_user_from_project", description="This api takes the project id and email to be deleted. DO NOT DELETE THE Owner through this route  ")
async def  delete_user_from_projects( project_id: str,  email: str, ):

    try:
    
        await delete_user_from_project( project_id= project_id, email= email )

        return {"detail": " user deleted.",
                    "project_id": project_id,
                    "email" : email
        }
    
                    
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")  
    

 #$~ API 17 ~$############################################################################################################################
#$~ Description ~$#
"""
gets users associated with project 

"""

@app.post("/get_users_for_project", description="This api takes the project id and returns the users associated with the project ")
async def  get_users_for_project( project_id: str):

    try:
    
        details = await  get_project_details_by_id( project_id= project_id)
        emails_list = json.loads(details['emails'])
        

        return {"users": emails_list}
    
                    
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")  
    

 #$~ API 18 ~$############################################################################################################################
#$~ Description ~$#
"""
gets users associated with project 

"""

@app.post("/get_executive_summary", description="This api takes the project id and email  and returns the executive summary as a pdf. filter based on roles will be added later")
async def  get_executive_summmary_for_project( project_id: str, email:str, project_name):

    try:
    
        details = await  get_executive_summary_from_db( project_id= project_id, email = email)
        details_parsed = json.loads(details)

        pdf = PDF()
        
        pdf.add_page()
        pdf.project_header(project_name)

        for section, markdown_content in details_parsed.items():
            pdf.set_font('Arial', 'B', 15)
            pdf.cell(0, 10, section, ln=True)
            pdf.ln(2)

            clean_markdown = clean_markdown_codeblock(markdown_content)

            pdf.markdown_to_pdf(clean_markdown)
            pdf.ln(3)

        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=executive_summary_{project_id}.pdf"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



#$~ API 19 ~$############################################################################################################################
#$~ Description ~$#
"""
create a tasklist  

"""

@app.post("/assistant_function_interact", description="This api takes the project id CHECKLIST_ASSISTANT ")
async def  create_tasklist_for_project( project_id: str, assistant_action: str, content: str):

    try:
    
        response = await  assistant.assistant_function(project_id= project_id, assistant_action =  assistant_action, content = content )
       
        return  response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

#$~ API 20 ~$############################################################################################################################
#$~ Description ~$#
"""
route to update a codebase.
"""

#$~ Task List ~$#
"""
1) Needs error handling on no files found


"""

@app.post("/updatecodebase", description=", Takes in project id and codebase, branch Id and updates it ")
async def update_codebase( file: UploadFile, project_id: str,  file_source :str , commit_id: str, background_tasks: BackgroundTasks):

    
    # Check if the codebase uploaded is actually a zip
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a zip file.")
    
    # Temporary directory created in the file system and deleted later
    temp_dir = tempfile.mkdtemp()
    project_details = await get_project_details_by_id(project_id)
    
    # Create initial entry in database asynchronously
    await update_status_in_db(emails = project_details["emails"], project_id=project_id, project_description = project_details["project_description"],   project_name = project_details["project_name"],  status="Project being updated", summary = None, executive_summary= None, project_diagrams = None, file_source = file_source, commit_id = commit_id)
    try:
        zip_path = os.path.join(temp_dir, file.filename)

        # Save the uploaded zip file
        async with aiofiles.open(zip_path, 'wb') as f:  
            contents = await file.read()  
            await f.write(contents)  

        # Extract the contents of the zip file
        extract_dir = os.path.join(temp_dir, 'extracted_files')
        os.makedirs(extract_dir)

        ### Helper for Zip Extraction
        def extract_zip_file(zip_path: str, extract_to: str):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
             zip_ref.extractall(extract_to)

        loop = asyncio.get_event_loop()  # Acquire the current event loop
        await loop.run_in_executor(None, extract_zip_file, zip_path, extract_dir) 


        # Process files in the background
        background_tasks.add_task(update_process_and_post_summary, extract_dir, project_details= project_details, project_id= project_id, file_source = file_source, commit_id = commit_id)
        
        return {"detail": "File upload successful, generating description in background.",
                "project_id": project_id,
                "project_name" :project_details["project_name"]}
    
    except Exception as e:
        shutil.rmtree(temp_dir)
        await update_status_in_db(emails = project_details["emails"], project_id=project_id, project_description = project_details["project_description"],   project_name = project_details["project_name"] , status = f"Project update failed: {str(e)}" , summary = None, executive_summary= None, project_diagrams = None,  file_source = file_source, commit_id = commit_id)
        raise HTTPException(status_code=500, detail="Internal server error")
    

# Iterate through each file in zip and call Open AI to generate the summary 

#`````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
def extract_first_json(text):
    """Extracts the first JSON object found in a string (non-greedy)."""
    match = re.search(r'\{(?:[^{}]|(?R))*\}', text)
    if match:
        return match.group(0)
    return None

def safe_json_loads(possible_json):
    """Try to load JSON, and extract if embedded or duplicated."""
    try:
        return json.loads(possible_json)
    except Exception:
        extracted = extract_first_json(possible_json)
        if extracted:
            try:
                return json.loads(extracted)
            except Exception:
                pass
        # As a last resort:
        print("Failed to parse summary as JSON:\n", possible_json)
        raise


async def process_file(file_path, extract_dir):
    relative_path = os.path.relpath(file_path, extract_dir)

    if should_ignore_path(relative_path):
        return None, None  # Skip ignored files

    if os.path.splitext(file_path)[1].lower() in TEXT_FILE_EXTENSIONS:
        file_path_1, content, summary = await summarize_file(file_path)
        print(f'Summary succeeded for {file_path}')
        summary_dict = safe_json_loads(summary)

        qualitative_score = summary_dict.get('qualitative_score', 0)
        summary = summary_dict.get('summary', '')

        if summary:
            file_name = os.path.basename(file_path)
            context_summary = {
                'score': qualitative_score,
                'text': f"file name is {file_name}, \nfile path is: {file_path_1}, \nfilesummary is: {summary}"
            }
            full_summary = [file_name, file_path, content, summary]
            return context_summary, full_summary

    return None, None

def should_ignore_path(path, ignore_patterns=DEFAULT_IGNORE_PATTERNS):
    path_parts = path.split(os.sep)
    for pattern in ignore_patterns:
        if any(fnmatch.fnmatch(part, pattern) for part in path_parts):
            return True
    return False

async def update_process_and_post_summary(extract_dir: str, project_id: str, project_details: str, file_source :str, commit_id :str):
    try:
        await update_status_in_db(emails=project_details["emails"], project_id=project_id, 
                                  project_description=project_details["project_description"],
                                  project_name=project_details["project_name"],
                                  status="Updated Summary being generated", summary=None, executive_summary=None, 
                                  project_diagrams=None, file_source=file_source, commit_id=commit_id)

        # Gather all file paths first
        all_file_paths = []
        for root, dirs, files in os.walk(extract_dir):
            dirs[:] = [d for d in dirs if not should_ignore_path(os.path.join(root, d), DEFAULT_IGNORE_PATTERNS)]

            for name in files:
                file_path = os.path.join(root, name)
                if os.path.splitext(file_path)[1].lower() in TEXT_FILE_EXTENSIONS and not should_ignore_path(file_path):
                    all_file_paths.append(file_path)

        # Process all files in parallel
        async def process_files_in_parallel(file_paths, batch_size=10):
            context_summaries = []
            full_summaries = []

            for i in range(0, len(file_paths), batch_size):
                batch = file_paths[i:i+batch_size]
                tasks = [process_file(file_path, extract_dir) for file_path in batch]
                results = await asyncio.gather(*tasks)

                for context_summary, full_summary in results:
                    if context_summary and full_summary:
                        context_summaries.append(context_summary)
                        full_summaries.append(full_summary)

            context_summaries.sort(key=lambda x: x['score'], reverse=True)
            combined_summary = '\n\n\n'.join([item['text'] for item in context_summaries])

            return combined_summary, full_summaries

        context_summaries, full_summaries = await process_files_in_parallel(all_file_paths)

        if context_summaries:
            combined_summary = anthropic_truncator(text=context_summaries)

            await update_vectors(project_id=project_id, full_summaries=full_summaries, action="update")
            executive_summary = await generate_executive_summary(combined_summary)
            diagrams = await generate_project_diagrams(project_id=project_id, summary=combined_summary)

            await store_summary_in_db(emails=project_details["emails"], project_id=project_id, summary=combined_summary,
                                      status="Updated", executive_summary=executive_summary, project_diagrams=diagrams)

            email_summary(combined_summary, project_details["emails"], project_id, project_details["project_name"])
            print("email sent")
        else:
            await update_status_in_db(emails=project_details["emails"], project_id=project_id, 
                                      project_description=project_details["project_description"],
                                      project_name=project_details["project_name"],
                                      status="codebase update failed, Contact: sai_002@harmonyengine.ai", 
                                      summary=None, executive_summary=None, project_diagrams=None, 
                                      file_source=file_source, commit_id=commit_id)
            print("no files were found")
    except Exception as e:
        error_msg = f"Error in process_and_post_summary: {str(e)}"
        print(error_msg)
        await update_status_in_db(emails=project_details["emails"], project_id=project_id, 
                                  project_description=project_details["project_description"],
                                  project_name=project_details["project_name"],
                                  status=error_msg, summary=None, executive_summary=None, project_diagrams=None, 
                                  file_source=file_source, commit_id=commit_id)
        await email_analysis_failed(project_details["emails"], project_details["project_name"])

    finally:
        shutil.rmtree(extract_dir)


#$~ API 21 ~$############################################################################################################################
#$~ Description ~$#
"""
reviews and provides ticket analysis  

"""

@app.post("/ticket_review", description="This api takes the project id and the ticket to return analysis to the call back url")
async def create_tasklist_for_project(req: TicketReviewRequest, background_tasks: BackgroundTasks):
    # Immediately acknowledge receipt with ticket_id
    background_tasks.add_task(
        process_and_callback,
        req.project_id,
        req.ticket_content,
        req.callback_url,
        req.ticket_id
    )
    return {
        "detail": "Request accepted.",
        "ticket_id": req.ticket_id
    }

async def process_and_callback(project_id, ticket_content, callback_url, ticket_id):
    try:
        response = await ticket_assistant(project_id=project_id, ticket_info=ticket_content)
        
        # Always send ticket_id in the callback payload
        result = {
            "ticket_id": ticket_id,
            "result": response
        }
        async with httpx.AsyncClient(timeout=120) as client:
            await client.post(callback_url, json=result)
    except Exception as exc:
        err_data = {"ticket_id": ticket_id, "error": str(exc)}
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                await client.post(callback_url, json=err_data)
        except Exception as e:
            error_msg = f"Error in process_and_post_summary: {str(e)}"
            print(error_msg)


#$~ API 22 ~$############################################################################################################################
#$~ Description ~$#

@app.post("/upload_file", description="Test API")
async def upload_file(file: UploadFile):
    try:
        IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
        EXCEL_EXTENSIONS = {".xlsx", ".xls", ".csv"}

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in IMAGE_EXTENSIONS.union(EXCEL_EXTENSIONS):
            raise HTTPException(status_code=400, detail="Invalid file type. Only images or Excel files allowed.")
        # Save the uploaded file to a temporary location
        content = await file.read()

        if ext.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
            mime_type = file.content_type
            base64_str = base64.b64encode(content).decode('utf-8')
            data_uri = f"data:{mime_type};base64,{base64_str}"
            response = await chat_pro.analyze_image(data_uri)

        elif ext in {".xlsx", ".xls", ".csv"}:
            if file.filename.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(content))
            else:
                df = pd.read_excel(io.BytesIO(content))
            excelText = df.to_string(index=False)
            response = await chat_pro.analyze_csv_or_excel(excelText)

        return JSONResponse(content={"summary": f"Summary of the file is: {response}"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

#$~ API 23 ~$############################################################################################################################
#$~ Description ~$#
"""
PR Review 
"""

@app.post("/pr_review_agent", description="Takes the git diff directly and post the summary")
async def create_tasklist_for_project(request: PRReviewRequest):
    response = await pr_review(
        project_id=request.project_id, 
        deep_review=request.deep_review, 
        git_diff=request.git_diff
    )
    return {
        "review": response
    }

############################################################################################################################

combined_app.mount("/", app)        # main app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(combined_app, host="0.0.0.0", port=8000)
