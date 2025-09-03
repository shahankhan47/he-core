from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from io import BytesIO
import re
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
import smtplib
from email import encoders
import os
import time
from openai import AsyncOpenAI
import asyncio
import random
import anthropic
import json
from typing import Dict, Tuple
import tiktoken
import chromadb
from chat.mermaid import generate_diagrams
import chardet
import textwrap
from db_operations import insert_or_update_summary_in_context_summaries
from .open_ai_token_counter import open_ai_truncator



# Configure API keys
api_key = os.getenv('OPENAI_API_KEY')
open_ai_client = AsyncOpenAI(api_key = api_key)

client = anthropic.Anthropic()
client.api_key = os.getenv('ANTHROPIC_API_KEY')
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")


# Email configuration for fast mail
sender_email = "sai_002@harmonyengine.ai"
smtp_server = "smtp.fastmail.com"
smtp_port = 465  
sender_password = os.environ.get('EMAIL_PASSWORD')

chroma_host = os.getenv("CHROMA_HOST", "https://chroma.agreeablesky-9ecb3d35.eastus.azurecontainerapps.io")
chroma_port = os.getenv("CHROMA_PORT", 8000)



class AsyncRateLimiter:
    def __init__(self, rate_limit):
        self.rate_limit = rate_limit
        self.tokens = rate_limit
        self.updated_at = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            while self.tokens < 1:
                self.add_new_tokens()
                await asyncio.sleep(0.1)
            self.tokens -= 1

    def add_new_tokens(self):
        now = time.monotonic()
        time_since_update = now - self.updated_at
        new_tokens = time_since_update * self.rate_limit
        if new_tokens > 1:
            self.tokens = min(self.tokens + new_tokens, self.rate_limit)
            self.updated_at = now
 
 
 
rate_limiter = AsyncRateLimiter(150)




async def update_vectors (project_id: str, full_summaries: list , action: None):

    # Initialize Chroma client
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

    # Initialize openai embedding function 
    import chromadb.utils.embedding_functions as embedding_functions

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key= api_key,
                model_name="text-embedding-3-large"
            )

    if action == "update":
        client.delete_collection(project_id)
    # Create a new collection
    
    collection = client.create_collection(
        name =  project_id,
        embedding_function = openai_ef
    )
    await insert_or_update_summary_in_context_summaries(project_id= project_id, full_summaries = str(full_summaries))

    for full_summary in full_summaries:

        file_name, file_path, content, summary = full_summary
        truncated_content = await open_ai_truncator(text= content, model= "text-embedding-3-large" , max_tokens= 5500)
        
        # Create the document with the required sections
        document = f"File Path: {file_path}\n\nSummary:\n{summary}\nContent:\n{truncated_content}"
        
        # Add the document to the collection
        try:
            collection.add(
                documents = document,
                ids= file_path.split('extracted_files/')[-1] if 'extracted_files/' in file_path else file_path,
                metadatas=[{"file_name": file_name}]
            )
            print(f'document added : { file_path}')
            
        
        except Exception as e:
             print(f"Error adding document for file '{file_name}': {e}")
    return None

async def summarize_with_openai(file_content: str, path: str) -> str:  
    retry_attempts = 10
    wait_times = [10, 30, 60, 360, 420, 480, 540, 600, 660, 720] 
    for attempt in range(retry_attempts):
        try:
            await rate_limiter.acquire()  
            truncted_text = await open_ai_truncator(text= file_content, model= "gpt-4o-mini" , max_tokens= 90000)

            response = await open_ai_client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {"role": "system", "content": """You are an expert software Developer who will generate summaries of code files from a larger codebase. 
                                                   Steps to perform, 1) Identify the nature of file and how it could relate to a larger codebase 
                                                                     2) If the file is a configuration or boilerplate code file that does not contain core business logic or code logic, generate a extreamly short and succint summary of 2 lines
                     .                                               3) generate a qualitative score from 1 to 5. 5 means a file has a high context regarding business code or logic.   value of  1 means the file is just a config type, boiler plate code. Eg a launch.js file 
                                                                     3) If the file contains code or documentation related to the core features of the codebase, Summarize the identified Features and Business context. 
                                                                     4) Include any dependencies that may depend on other files databases, tables etc. 
                                                                     5) come up with a summary of  bussiness logic or details of the logic in the function, the higher the score, the more detailed you can be. 
                                                                     """},
                    {"role": "user", "content": f"Analyze the following file. Do not include triple quotes or any escape sequences in the result.\n\n{path}.  \n\n{truncted_text} "}
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "code_summarizer",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "summary": {
                                    "type": "string"
                                },
                                "qualitative_score": {
                                    "type": "string"
                                },
                                
                            },
                            "required": ["summary", "qualitative_score"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                },
    
                max_output_tokens = 1100
            )
            return response.output_text
        
        except Exception as e:
            if attempt < retry_attempts - 1:
                wait_time = wait_times[attempt]
                print(f"OpenAI API error, retrying (attempt {attempt + 1}) after {wait_time} seconds: {str(e)}")
                await asyncio.sleep(wait_time)
            else:
                return f"Error in OpenAI API: {str(e)}"

def read_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            # Detect encoding
            result = chardet.detect(raw_data)
            encoding = result['encoding']

            # If encoding is None or binary file, return an error
            if encoding is None:
                print(f"Failed to detect encoding for {file_path}.")
                return ""
        
            try:
                return raw_data.decode(encoding)
            
            except Exception as decode_error:
                print(f"Decoding error for {file_path}: {decode_error}")
                return ""
    except Exception as e:
        print(f"Error opening file {file_path}: {e}")
        return ""      

async def summarize_file(file_path):

    content = read_file(file_path)
    if content:
        print(f"Summarizingfile:  {file_path}")
        file_path = re.sub(r"/.*/extracted_files/", "", file_path)
                
    return file_path, content,  await summarize_with_openai(content, path= file_path)
    





#email used in api 1 
def email_summary(summary: str, recipient_emails: str, project_id: str,  project_name: str ):
    try:
         
        email_list = json.loads(recipient_emails)

        for email_info in email_list:
                email = email_info['email']
                # Create the email message
                message = MIMEMultipart("alternative")
                message["From"] = '"Harmony Engine Support" <support@harmonyengine.ai>'
                message["To"] = email
                message["Subject"] = f"Codebase Analysis Complete - {project_name}"

                # Add a brief message to the email body
                body = textwrap.dedent(f"""
                Hello,
        
                Your analysis for project "{project_name}" is now complete.
        
                You can view the complete analysis by logging into your account at https://app.harmonyengine.ai/

                Need help? Contact our support team at support@harmonyengine.ai
        
                Best regards,  
                The Harmony Engine Team""")

                html = f"""\
                <div style="font-family: Arial, sans-serif; max-width: 600px; margin: auto; padding: 20px; background-color: #f9f9f9; border-radius: 10px; text-align: center;">
                <h2 style="color: #333;">✅ Codebase Analysis Complete</h2>
                <p style="color: #555;">Your analysis is now complete for the project:</p>
                <div style="font-size: 28px; font-weight: 700; color: #1a237e; background: #dbeafe; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    {project_name}
                </div>

                <p style="color: #999; font-size: 14px;">You can view the complete analysis by logging into your account at <a href="https://app.harmonyengine.ai/" style="color: #007BFF;">https://app.harmonyengine.ai/</a></p>
                
                <hr style="border: 0; border-top: 1px solid #ddd; margin: 20px 0;">
                
                <p style="font-size: 12px; color: #777;">Need help? Contact our support team at <a href="mailto:support@harmonyengine.ai" style="color: #007BFF;">support@harmonyengine.ai</a></p>
                
                <p style="font-size: 12px; color: #aaa;">Harmony Engine • Your AI-powered assistant</p>
                </div>
                """
                message.attach(MIMEText(body, "plain"))
                message.attach(MIMEText(html, "html"))
                # Connect to the SMTP server and send the email
                with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                    server.login(sender_email, sender_password)
                    server.send_message(message)
        return "emails sent "
                

    except Exception as e:
        print(f"Failed to send email: {str(e)}")

# Email when something goes wrong
def email_analysis_failed(recipient_emails: str,  project_name: str):
    try:
         
        email_list = json.loads(recipient_emails)

        for email_info in email_list:
                email = email_info['email']
                # Create the email message
                message = MIMEMultipart("alternative")
                message["From"] = '"Harmony Engine Support" <support@harmonyengine.ai>'
                message["To"] = email
                message["Subject"] = f"Codebase Analysis Failed - {project_name}"

                # Add a brief message to the email body
                body = textwrap.dedent(f"""
                Hello,
        
                Your analysis for project "{project_name}" has failed.
                Please delete the project and try again.
        
                You can view the complete analysis by logging into your account at https://app.harmonyengine.ai/

                Need help? Contact our support team at support@harmonyengine.ai
        
                Best regards,  
                The Harmony Engine Team""")

                html = f"""\
                <div style="font-family: Arial, sans-serif; max-width: 600px; margin: auto; padding: 20px; background-color: #f9f9f9; border-radius: 10px; text-align: center;">
                <h2 style="color: #333;">❌ Codebase Analysis Failed</h2>
                <p style="color: #555;">Your analysis has failed for the project:</p>
                <div style="font-size: 28px; font-weight: 700; color: #1a237e; background: #dbeafe; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    {project_name}
                </div>

                <p style="color: #999; font-size: 14px;">Please delete the project and try again</p>

                <p style="color: #999; font-size: 14px;">You can view the complete analysis by logging into your account at <a href="https://app.harmonyengine.ai/" style="color: #007BFF;">https://app.harmonyengine.ai/</a></p>
                
                <hr style="border: 0; border-top: 1px solid #ddd; margin: 20px 0;">
                
                <p style="font-size: 12px; color: #777;">Need help? Contact our support team at <a href="mailto:support@harmonyengine.ai" style="color: #007BFF;">support@harmonyengine.ai</a></p>
                
                <p style="font-size: 12px; color: #aaa;">Harmony Engine • Your AI-powered assistant</p>
                </div>
                """
                message.attach(MIMEText(body, "plain"))
                message.attach(MIMEText(html, "html"))
                # Connect to the SMTP server and send the email
                with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                    server.login(sender_email, sender_password)
                    server.send_message(message)
        return "emails sent "
                

    except Exception as e:
        print(f"Failed to send email: {str(e)}")

#Generating the executive summary
async def generate_executive_summary(summary: str) -> Dict[str, str]:
    questions_dict = {
        "Executive Summary": "Based on the provided summary, generate an summary that  captures the key business features of the code aimed at executives.",
        "Requirements and Setup Details": "Based on the provided summary, identify the requirements, code language, and any other setup details.",
        "User Personas": "Based on the provided summary, identify potential User Personas of the application.",
        "User Stories": "Based on the provided summary, identify the User Stories of this application.",
        "Key Business Modules": "Based on the provided summary, identify the Key Business Modules.",
        "Current Tech Stack": "Based on the provided summary, identify the Key Tools, Technologies and frame works being used",
        "Recommended Stable Tech Stack": "Based on the provided summary, identify if the code base can be updated to the most stable tech stack along with the justifications.",
        

    }
    
    retry_attempts = 5
    initial_delay = 1  # Initial delay in seconds

    # Initialize a dictionary to store responses
    responses = {}

    try:
        # Loop through each question in the dictionary and make API call
        for title, question in questions_dict.items():
            for attempt in range(retry_attempts):
                try:
                    response = client.messages.create(
                        model="claude-3-5-sonnet-20240620",
                        max_tokens=2000,
                        system=[
                            {"type": "text", "text": "You are an AI assistant tasked with analyzing codebases. Format your responses in markdown"},
                            {"type": "text", "text": f"Here is the summary of a complex codebase: {summary}", "cache_control": {"type": "ephemeral"}}
                        ],
                        messages=[{"role": "user", "content": question}]
                    )
                    print(response.usage)

                    # Store the response content in the responses dictionary with the corresponding title
                    responses[title] = response.content[0].text
                    break  # Exit the retry loop if the request is successful
                
                except Exception as e:
                    if attempt < retry_attempts - 1:  # If it's not the last attempt
                        sleep_time = initial_delay * (3 ** attempt)  # Exponential backoff
                        print(f"API Anthropic sleeping for {sleep_time} seconds on attempt {attempt + 1}. Error: {str(e)}")
                        time.sleep(sleep_time)
                    else:
                        responses[title] = f"Error in Anthropic API for question '{question}': {str(e)}"

    except Exception as e:
        print(f"Failed to process requests through Anthropic: {e}")

    return json.dumps(responses, indent=4)
    
async def generate_project_diagrams( project_id, summary):

    questions = {
            "Program Structure": "Generate a detailed project structure diagram of the codebase it should show the various components of the system",
            "User Flow": "Generate a detailed user flow diagram for this project",
            "User Personas": "Generate a detailed  user journey flow for this project"
        }
    try: 
    # Initialize a dictionary to store the results
        diagram_results = {}
        for key, question in questions.items():
            # Call the generate_diagrams function for each question
            diagram_result = await  generate_diagrams( project_id, question, summary)
            
            # Append the result to the dictionary with the original key
            diagram_results[key] = diagram_result
        return json.dumps(diagram_results, indent=4)     
    except Exception as e:
        print(f"Failed to generate mermaid: {str(e)}")
        
    



#############  api 2   ################################  
#Depricated 
# email used in stage 2 
def send_pdf_email(recipient_email, pdf_data):
    # Create the email object

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = "Harmony engine - Summary"

    # Email body
    body = """The Summary of the codebase has been enclosed in this email.
               
            Cheers!
            The Harmony Engine Team
            """
    msg.attach(MIMEText(body, 'plain'))

    # Attach the PDF
    attachment = MIMEBase('application', 'octet-stream')
    attachment.set_payload(pdf_data)
    encoders.encode_base64(attachment)
    attachment.add_header('Content-Disposition', 'attachment; filename="summary.pdf"')
    msg.attach(attachment)

    # Send the email
    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
            print("Stage 2 - PDF Email sent successfully.")
    except Exception as e:
        print(f"Failed to send PDF  email: {e}")  

def create_pdf(summary, qa_pairs):
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))

    story = []

    # Add summary
     # Ensure summary is a string
    if isinstance(summary, list) and summary and hasattr(summary[0], 'text'):
        print("Extracting text from the first element in summary list")
        summary = summary[0].text
    elif not isinstance(summary, str):
        raise ValueError("Summary should be a string or a list containing an object with a 'text' attribute")
    

    story.append(Paragraph("Summary", styles['Heading1']))
    story.append(Paragraph(summary, styles['Justify']))
    story.append(Spacer(1, 12))
    
       # Ensure qa_pairs is a string
    if not isinstance(qa_pairs, str):
        raise ValueError("qa_pairs should be a string")
    
    # Split the qa_pairs string into individual Q&A pairs
    qa_list = re.split(r'Q:', qa_pairs)[1:]  # Skip the first empty element

    # Process each Q&A pair
    for qa in qa_list:
        try:
            # Split each pair into question and answer
            q, a = qa.split('A:', 1)
            
            question = q.strip()
            answer = a.strip()

            # Extract text from TextBlock if present
            text_block_match = re.search(r"TextBlock\(text=[\"'](.*?)[\"'],", answer, re.DOTALL)
            if text_block_match:
                answer = text_block_match.group(1)
            else:
                # If no TextBlock, clean up any remaining square brackets
                answer = re.sub(r'\[|\]', '', answer).strip()

            # Question in bold
            story.append(Paragraph(f"<b>Q: {question}</b>", styles['Normal']))
            story.append(Spacer(1, 6))
            
            # Answer as regular paragraph
            story.append(Paragraph(f"A: {answer}", styles['Justify']))
            story.append(Spacer(1, 12))

        except Exception as e:
            print(f"Error processing QA pair: {str(e)}")
            print(f"Problematic QA pair: {qa}")

    # Build the PDF
    doc.build(story)
    
    return buffer.getvalue()


def analyze_summary_with_anthropic(summary: str) -> Tuple[str, str]:
    questions = [
        "Based on the provided summary, come up with a list of high-level questions that capture the user stories and business features of the code.",
        "Based on the code produce an executive summary that captures the key business features of the code."
    ]
    
    retry_attempts = 5
    initial_delay = 10  # Initial delay in seconds

    try:
        # Initialize response variables
        questions_response = None
        executive_summary_response = None

    
        # Loop through each question and make API call
        for idx, user_question in enumerate(questions):
            for attempt in range(retry_attempts):
                try:
                    response = client.beta.prompt_caching.messages.create(
                        model="claude-3-5-sonnet-20240620",
                        max_tokens=2000,
                        system=[
                            {"type": "text", "text": "You are an AI assistant tasked with analyzing codebases."},
                            {"type": "text", "text": f"Here is the summary of a complex codebase: {summary}", "cache_control": {"type": "ephemeral"}}
                        ],
                        messages=[{"role": "user", "content": user_question}]
                    )
                    if idx == 0:
                        questions_response = response.content
            
                    elif idx == 1:
                        executive_summary_response = response.content
                    break  # Exit the retry loop if the request is successful
                except Exception as e:
                    if attempt < retry_attempts - 1:  # If it's not the last attempt
                        sleep_time = initial_delay * (3 ** attempt)  # Exponential backoff
                        print(f"API Anthropic sleeping for {sleep_time} seconds on attempt {attempt + 1}. for error {str(e)}")
                        time.sleep(sleep_time)
                    else:
                        error_message = f"Error in Anthropic API for question '{user_question}': {str(e)}"
                        if idx == 0:
                            questions_response = error_message
                        elif idx == 1:
                            executive_summary_response = error_message

    except Exception as e:
        print(f"Failed to create executive summary or questions, Anthropic: {e}")

    return questions_response, executive_summary_response

async def exponential_backoff(retries):
    return min(3 ** retries, 300)  # Cap at 60 seconds

async def generate_questions_openai(base_questions):
    retry_count = 0
    max_retries= 7
    while retry_count < max_retries:
        try:
            # Initialize the OpenAI thread
            run = await open_ai_client.beta.threads.create_and_run(
                assistant_id="asst_Xq0G9z95RI3Jl42dWzjloI7O",
                thread={
                    "messages": [
                        {"role": "user", "content": base_questions},
                    ]
                }
            )
            
            # Check the status of the thread until it's completed
            status = run.status
            while status != 'completed':
                await asyncio.sleep(5)
                run = await open_ai_client.beta.threads.runs.retrieve(
                    thread_id=run.thread_id,
                    run_id=run.id
                )
                status =  run.status

            if status == 'completed':
                thread = await open_ai_client.beta.threads.messages.list(run.thread_id)
                last_message = thread.data[0]
                data = json.loads(last_message.content[0].text.value)
                print("Last message content:", data)
                return data  # successfully completed

            else:
                print("Thread did not complete successfully.")
                return None

        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(f"Error: OpenAI thread failed after {max_retries} attempts. Last error: {str(e)}")
                return None  # or raise an exception if preferred

            await exponential_backoff(retry_count)
            print(f"Retrying... Attempt {retry_count + 1} after backoff")

async def generate_answers_anthropic(summary: str, custom_query: str) -> str:
    retry_attempts = 5
    initial_delay = 25  # Initial delay in seconds
    
    for attempt in range(retry_attempts):
        try:
            
            
            print("Analyzing summary with custom query:", custom_query)

            response = client.beta.prompt_caching.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2000,
                system=[
                    {"type": "text", "text": "You are an AI assistant tasked with analyzing codebases."},
                    {"type": "text", "text": f"Here is the summary of a complex codebase: {summary}", "cache_control": {"type": "ephemeral"}}
                ],
                messages=[{"role": "user", "content": custom_query}]
            )
            return response.content
        except anthropic.exceptions.RateLimitError as e:
            if attempt < retry_attempts - 1:  # If it's not the last attempt
                retry_after = int(e.response.headers.get('Retry-After', initial_delay))
                sleep_time = retry_after + random.uniform(0, 1)  # Add some jitter
                print(f"RateLimitError: Sleeping for {sleep_time} seconds")
                await asyncio.sleep(sleep_time) 
            else:
                return f"Error in Anthropic API: {str(e)}"
        except Exception as e:
            if attempt < retry_attempts - 1:  # If it's not the last attempt
                sleep_time = initial_delay * (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                print(f"Exception: {str(e)} - API 2 , Anthropic answer, Sleeping for {sleep_time} seconds")
                time.sleep(sleep_time)
            else:
                return f"Error in Anthropic API: {str(e)}"
    

async def generate_responses(final_question, email, summary_content, project_id, executive_summary):
    responses = []
    response_text = ""

    # update_status_in_db(email, project_id, "STAGE 2 Generating detailed responses")

    for category in final_question["questions"]:
        print(f"Processing Heading: {category['heading']}")

        for question in category["questions"]:
            print(f"Processing question: {question}")
            retries = 0
            while True:
                try:
                    await asyncio.sleep(5)
                    answer = await generate_answers_anthropic(summary_content, custom_query=question)
                    responses.append({"question": question, "answer": answer})
                    response_text += f"Q: {question}\nA: {answer}\n\n"
                    break  # Success, exit the retry loop
                except Exception as e:
                    retries += 1
                    if retries > 5:
                        # update_status_in_db(email, project_id, f"Error: STAGE 2 Failed to process question after {retries} attempts")
                        print(f"Failed after {retries} attempts: {str(e)}")
                        break
                    await exponential_backoff(retries)

    with open('responses.txt', 'w') as file:
        file.write(response_text)

    try:
        # update_status_in_db(email, project_id, "STAGE 2 Creating PDF")
        pdf_data = create_pdf(executive_summary, response_text )
        
        # update_status_in_db(email, project_id, "STAGE 2 Sending email")
        send_pdf_email(email, pdf_data)
        # update_status_in_db(email, project_id, "Ready")
        print("Email sent successfully")
    except Exception as e:
        # update_status_in_db(email, project_id, f"STAGE 2 Error: Failed to create PDF or send email - {str(e)}")
        print(f"Failed to create PDF or send email: {e}")



#API 3 
#Used to fetch the result of the computed summary 
