import asyncio
import asyncpg
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

# Database connection parameters
DB_PARAMS = {
    'database': os.environ.get('DB_NAME'),
    'user': os.environ.get('DB_USER'),
    'password': os.environ.get('DB_PASSWORD'),
    'host': os.environ.get('DB_HOST'),
    'port': os.environ.get('DB_PORT')
}

async def get_db_connection() -> asyncpg.Connection:
    return await asyncpg.connect(**DB_PARAMS)

async def get_summary_tables(conn: asyncpg.Connection) -> List[str]:
    query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_name LIKE 'summaries_%'
    """
    rows = await conn.fetch(query)
    return [row['table_name'] for row in rows]

async def extract_email_from_table_name(table_name: str) -> str:
    # Remove 'summaries_' prefix
    email_part = table_name[10:]
    # Reconstruct email
    return email_part

async def process_tables():
    conn = None
    try:
        conn = await get_db_connection()
        tables = await get_summary_tables(conn)

        for table_name in tables:
            email = await extract_email_from_table_name(table_name)
            
            # Get owner records from summary table
            query = f"""
                SELECT 
                    project_id,
                    project_name,
                    COALESCE(NULLIF(project_description, ''), ' ') as project_description,
                    created_at,
                    role
                FROM {table_name}
                WHERE role = 'owner'
            """
            owner_records = await conn.fetch(query)

            for record in owner_records:
                # Check if project exists in projects table
                exists_query = """
                    SELECT EXISTS(
                        SELECT 1 
                        FROM projects_table 
                        WHERE project_id = $1
                    )
                """
                project_exists = await conn.fetchval(exists_query, record['project_id'])

                if not project_exists:
                    # Insert new project
                    insert_query = """
                        INSERT INTO projects_table (
                            project_id,
                            owner_email,
                            email,
                            project_name,
                            project_description,
                            user_role,
                            created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """
                    await conn.execute(
                        insert_query,
                        record['project_id'],
                        email,
                        email,
                        record['project_name'],
                        record['project_description'],
                        record['role'],
                        record['created_at']
                    )
                    print(f"Inserted project {record['project_id']} from {email}")
                else:
                    print(f"Skipped project {record['project_id']} - already exists")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise e

    finally:
        if conn:
            await conn.close()

# Run the script
asyncio.run(process_tables())