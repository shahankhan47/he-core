
import asyncpg
from asyncpg import Connection, Record

from typing import Optional
import os
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Database connection parameters
DB_PARAMS = {
    'database': os.environ.get('DB_NAME'),    # should be 'database' instead of 'dbname'
    'user': os.environ.get('DB_USER'),
    'password': os.environ.get('DB_PASSWORD'),
    'host': os.environ.get('DB_HOST'),
    'port': os.environ.get('DB_PORT')
}

async def  get_db_connection():
    return await asyncpg.connect(**DB_PARAMS)

async def create_user_table(email: str):
    conn = None
    try:
        conn = await get_db_connection()
        
        # Create table for the user if not exists
        table_name = f"summaries_{email.replace('@', '_').replace('.', '_')}"
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name}(
                project_id VARCHAR PRIMARY KEY,
                status VARCHAR NOT NULL,
                project_name VARCHAR,
                project_description VARCHAR,
                role VARCHAR,
                summary TEXT,
                executive_summary TEXT,
                project_diagrams TEXT,
                file_source TEXT,
                commit_id VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        
        await conn.execute(create_table_query)
        print(f"Table created for email: {email}")

    except Exception as e:
        print(f"An error occurred while creating the table: {e}")
    
    finally:
        if conn:
            await conn.close()

async def update_status_in_db(emails: str, project_id: str, status: str,  project_name: str, project_description: str, summary: None , executive_summary: None,  project_diagrams: None, file_source: None, commit_id: None):
    
    try:
        conn = await get_db_connection()
        email_list = json.loads(emails)

        for email_info in email_list:
            email = email_info['email'].replace('@', '_').replace('.', '_')
            table_name = f"summaries_{email}"

        
            # Check if the table exists
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = $1
                )
            """, table_name)
        
            
            if not table_exists:
                    # If the table doesn't exist, create it
                    await create_user_table(email_info['email'])  
                
            # Proceed with the update/insert operation
            upsert_query = f"""
                INSERT INTO {table_name} 
                (project_id, status, project_name, project_description, role,  summary, executive_summary, project_diagrams, file_source, commit_id)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9, $10)
                ON CONFLICT (project_id)
                DO UPDATE SET 
                status = $2, 
                summary = $6, 
                executive_summary = $7, 
                project_diagrams = $8,
                file_source = $9,
                commit_id = $10
            """
            
            await conn.execute(upsert_query, project_id, status, project_name, project_description,  email_info['role'], summary, executive_summary, project_diagrams, file_source, commit_id)
            print(f"Status updated for email: {email_info['email']}, project_id: {project_id}")

    except Exception as e:
        print(f"An error occurred while updating the status: {e}")
    
    finally:
        if conn:
            await conn.close()


async def store_summary_in_db(emails: str, project_id: str, summary: str, status: str, executive_summary: str, project_diagrams: str):
    conn: Optional[Connection] = None
    try:
        conn = await get_db_connection()
        email_list = json.loads(emails)

        for email_info in email_list:
            email = email_info['email'].replace('@', '_').replace('.', '_')
            table_name = f"summaries_{email}"
            update_query = f"""
                UPDATE {table_name}
                SET summary = $2,
                status = $3,
                executive_summary = $4,
                project_diagrams = $5
                WHERE project_id = $1;
            """
            
            await conn.execute(update_query, project_id, summary, status, executive_summary, project_diagrams)

    except Exception as e:
        print(f"An error occurred while storing the summary: {e}")
    
    finally:
        if conn:
            await conn.close()

async def get_summary_from_db(email: str, project_id: str) -> Optional[str]:
    conn = None
    try:
        conn = await get_db_connection()
        
        table_name = f"summaries_{email.replace('@', '_').replace('.', '_')}"
        select_query = f"""
            SELECT summary FROM {table_name}
            WHERE project_id = $1
            ORDER BY created_at DESC
            LIMIT 1
        """
        
        result = await conn.fetchrow(select_query, project_id)
        return result['summary'] if result else None

    except Exception as e:
        print(f"An error occurred while retrieving the summary: {e}")
        return None
    finally:
        if conn:
            await conn.close()

async def get_executive_summary_from_db(email: str, project_id: str) -> Optional[str]:
    conn = None
    try:
        conn = await get_db_connection()
        
        table_name = f"summaries_{email.replace('@', '_').replace('.', '_')}"
        select_query = f"""
            SELECT  executive_summary FROM {table_name}
            WHERE project_id = $1
            ORDER BY created_at DESC
            LIMIT 1
        """
        
        result = await conn.fetchrow(select_query, project_id)
        return result['executive_summary'] if result else None

    except Exception as e:
        print(f"An error occurred while retrieving the executive_summary: {e}")
        return None
    
    finally:
        if conn:
            await conn.close()



async def get_project_diagrams_from_db(email: str, project_id: str) -> Optional[str]:
    conn = None
    try:
        conn = await get_db_connection()
        
        table_name = f"summaries_{email.replace('@', '_').replace('.', '_')}"
        select_query = f"""
            SELECT  project_diagrams FROM {table_name}
            WHERE project_id = $1
            ORDER BY created_at DESC
            LIMIT 1
        """
        
        result = await conn.fetchrow(select_query, project_id)
        return result['project_diagrams'] if result else None

    except Exception as e:
        print(f"An error occurred while retrieving the project_diagrams: {e}")
        return None
    
    finally:
        if conn:
            await conn.close()




async def create_user_conversation_table(email: str):
    conn = None
    try:
        conn = await get_db_connection()
        
        table_name = f"conversations_{email.replace('@', '_').replace('.', '_')}"
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                project_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        
        await conn.execute(create_table_query)

    except Exception as e:
        print(f"An error occurred while creating the table: {e}")
    finally:
        if conn:
            await conn.close()

async def store_conversation_in_db(email: str, project_id: str, role: str, content: str):
    conn = None
    try:
        conn = await get_db_connection()
        
        await create_user_conversation_table(email)
        
        table_name = f"conversations_{email.replace('@', '_').replace('.', '_')}"
        insert_query = f"""
            INSERT INTO {table_name} (project_id, role, content)
            VALUES ($1, $2, $3)
        """
        
        await conn.execute(insert_query, project_id, role, content)

    except Exception as e:
        print(f"An error occurred while storing the conversation: {e}")
    finally:
        if conn:
            await conn.close()


async def get_conversation_history_from_db(email: str, project_id: str) -> list[dict[str, str]]:
    conn = None
    try:
        conn = await get_db_connection()
        
        table_name = f"conversations_{email.replace('@', '_').replace('.', '_')}"
        
        # Check if the table exists
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = $1
            )
        """, table_name)
        
        if not table_exists:
            # If the table doesn't exist, create it
            await create_user_conversation_table(email)


        select_query = f"""
            WITH ranked_messages AS (
                SELECT role, content, created_at,
                       ROW_NUMBER() OVER (ORDER BY created_at DESC) as row_num
                FROM {table_name}
                WHERE project_id = $1
            )
            SELECT role, content
            FROM ranked_messages
            WHERE row_num <= 10
            ORDER BY created_at ASC
        """
        
        results = await conn.fetch(select_query, project_id)
        return [{"role": row["role"], "content": row["content"]} for row in results]

    except Exception as e:
        print(f"An error occurred while retrieving the conversation: {e}")
        return []
    finally:
        if conn:
            await conn.close()

async def get_user_projects(email: str) -> str:
    conn = None
    try:
        conn = await get_db_connection()
        
        table_name = f"summaries_{email.replace('@', '_').replace('.', '_')}"
        
        check_table_query = f"""
            SELECT EXISTS (
                SELECT  FROM information_schema.tables 
                WHERE table_name = $1
            )
        """
        
        table_exists = await conn.fetchval(check_table_query, table_name)
        
        if not table_exists:
            return json.dumps({"projects": []})

        select_query = f"""
            SELECT DISTINCT ON (project_id) project_id, created_at, status, project_name, role, project_description, file_source, commit_id 
            FROM {table_name}
            ORDER BY project_id, created_at DESC
        """
        
        results = await conn.fetch(select_query)
        
        projects = [
            {
                "project_name": row["project_name"],
                "project_id": row["project_id"],
                "created_at": row["created_at"].isoformat() if isinstance(row["created_at"], datetime) else str(row["created_at"]),
                "status": row["status"],
                "project_description": row["project_description"],
                "role": row["role"],
                "file_source": row["file_source"],
                "commit_id": row["commit_id"],
            }
            for row in results
        ]
        
        return json.dumps({"projects": projects})

    except Exception as e:
        print(f"An error occurred while retrieving user projects: {e}")
        return json.dumps({"projects": [], "error": str(e)})
    finally:
        if conn:
            await conn.close()



async def delete_project_data(email: str, project_id: str):
    conn = None
    
    try:
        conn = await get_db_connection()
        details = await get_project_details_by_id(project_id)
        email_list = json.loads(details['emails'])

        # Extract emails using for loop
        for item in email_list:
            email = item['email']
            role = item['role']
            # Table names
            
            summaries_table_name = f"summaries_{email.replace('@', '_').replace('.', '_')}"
            conversations_table_name = f"conversations_{email.replace('@', '_').replace('.', '_')}"
            files_table_name = f"uploaded_files_{email.replace('@', '_').replace('.', '_')}"
            
            # Check if summaries table exists
            summaries_table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = $1
                )
            """, summaries_table_name)
            
            if summaries_table_exists:
                delete_summaries_query = f"""
                    DELETE FROM {summaries_table_name}
                    WHERE project_id = $1
                """
                await conn.execute(delete_summaries_query, project_id)
                print(f"Deleted summaries for project_id: {project_id}")
            else:
                print(f"Summaries table does not exist for email: {email}")
            
            # Check if conversations table exists
            conversations_table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = $1
                )
            """, conversations_table_name)
            
            if conversations_table_exists:
                delete_conversations_query = f"""
                    DELETE FROM {conversations_table_name}
                    WHERE project_id = $1
                """
                await conn.execute(delete_conversations_query, project_id)
                print(f"Deleted conversations for project_id: {project_id}")
            else:
                print(f"Conversations table does not exist for email: {email}")

            # Check if summaries table exists
            files_table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = $1
                )
            """, files_table_name)
            
            if files_table_exists:
                delete_uploaded_files_query = f"""
                    DELETE FROM {files_table_name}
                    WHERE project_id = $1
                """
                await conn.execute(delete_uploaded_files_query, project_id)
                print(f"Deleted uploaded files metadata for project: {project_id}")
            else:
                print(f"File upload table does not exist for email: {email}")

            
            delete_assistants_thread_query = f"""
                DELETE FROM {"assistants_table"}
                WHERE project_id = $1
                """
            await conn.execute(delete_assistants_thread_query, project_id)
            print(f"Deleted all assistants thread for project_id: {project_id}")
         
            delete_project_details_query = f"""
                DELETE FROM {"projects_table"}
                WHERE project_id = $1
            """
            await conn.execute(delete_project_details_query, project_id)
            print(f"Deleted summaries from project table for project_id: {project_id}")
          
        
    except Exception as e:
        print(f"An error occurred while deleting project data: {e}")
    finally:
        if conn:
            await conn.close()



async def create_user_pin_table(email: str):
    conn = None
    try:
        conn = await get_db_connection()
        
        table_name = f"pins_{email.replace('@', '_').replace('.', '_')}"
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                project_id TEXT NOT NULL,
                topic_name TEXT NOT NULL,
                pin_content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        
        await conn.execute(create_table_query)

    except Exception as e:
        print(f"An error occurred while creating the pin table: {e}")
    finally:
        if conn:
            await conn.close()

async def create_pin_in_db(email: str, project_id: str, topic_name: str, pin_content: str):
    conn = None
    try:
        conn = await get_db_connection()
        
        await create_user_pin_table(email)
        
        table_name = f"pins_{email.replace('@', '_').replace('.', '_')}"
        insert_query = f"""
            INSERT INTO {table_name} (project_id, topic_name, pin_content)
            VALUES ($1, $2, $3)
        """
        
        await conn.execute(insert_query, project_id, topic_name, pin_content)

    except Exception as e:
        print(f"An error occurred while storing the pin: {e}")
    finally:
        if conn:
            await conn.close()


async def delete_pin_from_db(email: str, pin_id: int):
    conn = None
    try:
        conn = await get_db_connection()
        
        # Construct the table name using the user's email
        table_name = f"pins_{email.replace('@', '_').replace('.', '_')}"
        
        # Prepare the delete query for the specific pin by id
        delete_query = f"DELETE FROM {table_name} WHERE id = $1"
        
        # Execute the delete query
        result = await conn.execute(delete_query, pin_id)
        
        # Check if a record was deleted
        if result == "DELETE 0":
            print(f"No pin with id {pin_id} found for user {email}")
            raise Exception(f"No pin with id {pin_id} found for user {email}")

    except Exception as e:
        print(f"An error occurred while deleting the pin: {e}")
    finally:
        if conn:
            await conn.close()

async def get_pins_from_db(email: str, project_id: str):
    conn = None
    try:
        conn = await get_db_connection()
        
        # Construct the table name using the user's email
        table_name = f"pins_{email.replace('@', '_').replace('.', '_')}"
        
        # Prepare the select query to fetch pins based on project_id
        select_query = f"SELECT * FROM {table_name} WHERE project_id = $1"
        
        # Execute the query and fetch results
        pins = await conn.fetch(select_query, project_id)
        
        return pins

    except Exception as e:
        print(f"An error occurred while fetching the pins: {e}")
        return None
    finally:
        if conn:
            await conn.close()



async def create_project_table():
    conn = None
    try:
        conn = await get_db_connection()
        
        table_name = f"projects_table"
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                project_id TEXT NOT NULL ,
                owner_email TEXT NOT NULL,
                email TEXT NOT NULL,
                project_name TEXT NOT NULL,
                project_description TEXT NOT NULL,
                user_role TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        
        await conn.execute(create_table_query)

    except Exception as e:
        print(f"An error occurred while creating the pin table: {e}")
    finally:
        if conn:
            await conn.close()



async def update_project_in_db(project_id :str , owner_email :str, project_name :str, project_description :str, emails :str):
    conn = None
    try:
        conn = await get_db_connection()
        
        await create_project_table()
        
        table_name = f"projects_table"
        # Define the insert query
        insert_query = f"""
            INSERT INTO {table_name} ( project_id,  owner_email, email, project_name, project_description,  user_role)
            VALUES ($1, $2, $3, $4, $5, $6)
        """

        # Insert an entry for the owner
        await conn.execute(insert_query, project_id, owner_email, owner_email, project_name, project_description, "Owner" )

        # Iterate over the emails dictionary to insert entries for each email
        for email, role in emails.items():
            await conn.execute(insert_query, project_id, owner_email, email, project_name, project_description, role)

    except Exception as e:
        print(f"An error occurred while storing the pin: {e}")
    finally:
        if conn:
            await conn.close()


async def get_project_details_by_id(project_id):
    
    try:
        conn = await get_db_connection()  
        query = """
            SELECT 
                owner_email,
                project_name,
                project_description,
                json_agg(json_build_object('email', email, 'role', user_role)) AS emails
            FROM 
                projects_table
            WHERE 
                project_id = $1
            GROUP BY 
                owner_email, project_name, project_description;
        """
        
        result = await conn.fetchrow(query, project_id)
        
        if result:
            project_details = {
                "owner_email": result["owner_email"],
                "project_name": result["project_name"],
                "project_description": result["project_description"],
                "emails": result["emails"]
            }
            return project_details
        else:
            return None
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    finally:
        if conn:
            await conn.close()    


async def add_user_to_project(project_id, users, owner_email):
    try:
        conn = await get_db_connection()
        email_list = json.loads(users)
        table_name = f"summaries_{owner_email.replace('@', '_').replace('.', '_')}"
        
        query = f"""
            SELECT 
                project_name,
                project_description,
                status,
                summary,
                executive_summary,
                project_diagrams,
                created_at,
                file_source,
                commit_id     
            FROM 
                {table_name} 
            WHERE 
                project_id = $1
           
        """
        
        result = await conn.fetchrow(query, project_id)

        if not result:
            raise ValueError(f"Project with ID {project_id} not found")

        for email_info in email_list:
            email = email_info['email'].replace('@', '_').replace('.', '_')
            table_name = f"summaries_{email}"

        
            # Check if the table exists
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = $1
                )
            """, table_name)
        
            
            if not table_exists:
                    # If the table doesn't exist, create it
                    await create_user_table(email_info['email'])  
                
            # Proceed with the update/insert operation
            insert_query = f"""
                INSERT INTO {table_name} 
                (project_id, status, project_name, project_description, role,  summary, executive_summary, project_diagrams, created_at, file_source, commit_id)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                
            """
            
            await conn.execute(insert_query, project_id, result['status'], result['project_name'], result['project_description'],  email_info['role'], result['summary'], result['executive_summary'], result['project_diagrams'], result['created_at'], result['file_source'], result['commit_id'])
            print(f"Status updated for email: {email_info['email']}, project_id: {project_id}")
            project_description = ' ' if not result['project_description'] else result['project_description']

            table_name = f"projects_table"
            # Define the insert query
            insert_query = f"""
                INSERT INTO {table_name} ( project_id, owner_email, email, project_name, project_description,  user_role)
                VALUES ($1, $2, $3, $4, $5, $6)
            """

            # Insert an entry for user in the project table 
            await conn.execute(insert_query, project_id, owner_email, email_info['email'], result['project_name'],project_description,  email_info['role'] )

    except Exception as e:
        print(f"An error occurred while updating the status: {e}")
    
    finally:
        if conn:
            await conn.close()

async def delete_user_from_project(email: str, project_id: str):
    conn = None
    
    try:
        conn = await get_db_connection()
       
        summaries_table_name = f"summaries_{email.replace('@', '_').replace('.', '_')}"
        conversations_table_name = f"conversations_{email.replace('@', '_').replace('.', '_')}"
        
        # Check if summaries table exists
        summaries_table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = $1
            )
        """, summaries_table_name)
        
        if summaries_table_exists:
            delete_summaries_query = f"""
                DELETE FROM {summaries_table_name}
                WHERE project_id = $1
            """
            await conn.execute(delete_summaries_query, project_id)
            print(f"Deleted summaries for project_id: {project_id}")
        else:
            print(f"Summaries table does not exist for email: {email}")
        
        # Check if conversations table exists
        conversations_table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = $1
            )
        """, conversations_table_name)
        
        if conversations_table_exists:
            delete_conversations_query = f"""
                DELETE FROM {conversations_table_name}
                WHERE project_id = $1
            """
            await conn.execute(delete_conversations_query, project_id)
            print(f"Deleted conversations for project_id: {project_id}")
        else:
            print(f"Conversations table does not exist for email: {email}")

         
        delete_project_details_query = f"""
            DELETE FROM {"projects_table"}
            WHERE project_id = $1 AND email = $2
        """
        await conn.execute(delete_project_details_query, project_id, email)
        print(f"Deleted summaries from project table for project_id: {project_id}")
          
        
    except Exception as e:
        print(f"An error occurred while deleting project data: {e}")
    finally:
        if conn:
            await conn.close()

async def create_assistants_table():
    conn = None
    try:
        conn = await get_db_connection()
        
        create_table_query = """
            CREATE TABLE IF NOT EXISTS assistants_table (
                id SERIAL PRIMARY KEY,
                project_id TEXT NOT NULL,
                assistant_name TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (project_id, assistant_name)
            )
        """
        
        await conn.execute(create_table_query)

    except Exception as e:
        print(f"An error occurred while creating the assistants table: {e}")
    finally:
        if conn:
            await conn.close()

async def insert_new_thread(project_id: str, assistant_name: str, thread_id: str):
    conn = None
    try:
        conn = await get_db_connection()

        insert_query = """
            INSERT INTO assistants_table (project_id, assistant_name, thread_id)
            VALUES ($1, $2, $3)
            ON CONFLICT (project_id, assistant_name)
            DO NOTHING
        """

        await conn.execute(insert_query, project_id, assistant_name, thread_id)
        print("New thread inserted successfully.")

    except Exception as e:
        print(f"An error occurred while inserting the new thread: {e}")
    finally:
        if conn:
            await conn.close()

async def get_thread(project_id: str, assistant_name: str):
    conn = None
    try:
        conn = await get_db_connection()
        await create_assistants_table()

        query_check_thread = """
            SELECT thread_id FROM assistants_table 
            WHERE project_id = $1 AND assistant_name = $2
        """
        
        row = await conn.fetchrow(query_check_thread, project_id, assistant_name)
        
        if row and row['thread_id']:
            return row['thread_id']
        else:
            return "thread does not exist"

    except Exception as e:
        print(f"An error occurred while fetching the thread: {e}")
    finally:
        if conn:
            await conn.close()

async def update_thread_id(project_id: str, thread_id: str, assistant_name: str):
    conn = None
    try:
        conn = await get_db_connection()
        await create_assistants_table()

        query_update_thread = """
            UPDATE assistants_table
            SET thread_id = $1
            WHERE project_id = $2 AND assistant_name = $3
        """

        # Use 'execute' and then fetch the number of rows affected
        result = await conn.execute(query_update_thread, thread_id, project_id, assistant_name)

        # The execute result typically returns a string like 'UPDATE <number>'
        affected_rows = result.split()[-1]

        if affected_rows == '0':
            await insert_new_thread(project_id, assistant_name, thread_id)
            print("No existing thread. New thread created.")
        else:
            print("Thread ID updated successfully.")

    except Exception as e:
        print(f"An error occurred while updating the thread ID: {e}")
    finally:
        if conn:
            await conn.close()

async def ensure_context_summaries_table_exists(conn: Connection):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS context_summaries (
        project_id VARCHAR PRIMARY KEY,
        full_summaries TEXT
    );
    """
    await conn.execute(create_table_query)

async def insert_or_update_summary_in_context_summaries(project_id: str, full_summaries: str) -> None:
    conn: Optional[Connection] = None
    try:
        conn = await get_db_connection()
        
        # Ensure the table exists
        await ensure_context_summaries_table_exists(conn)
        
        # Use an upsert query to insert or update
        upsert_query = """
        INSERT INTO context_summaries (project_id, full_summaries)
        VALUES ($1, $2)
        ON CONFLICT (project_id) DO UPDATE 
        SET full_summaries = EXCLUDED.full_summaries;
        """
        
        # Execute the upsert query
        await conn.execute(upsert_query, project_id, full_summaries)

    except Exception as e:
        print(f"An error occurred while inserting or updating the summary: {e}")
    
    finally:
        if conn:
            await conn.close()