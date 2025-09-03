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

async def column_exists(conn: asyncpg.Connection, table_name: str, column_name: str) -> bool:
    query = f"""
        SELECT EXISTS (
            SELECT 1 
            FROM information_schema.columns 
            WHERE table_name='{table_name}' AND column_name='{column_name}'
        )
    """
    return await conn.fetchval(query)

async def update_null_roles():
    conn = None
    try:
        conn = await get_db_connection()
        tables = await get_summary_tables(conn)

        for table_name in tables:
            # Check if 'commit_id' column exists
            if await column_exists(conn, table_name, 'commit_id'):
                # Update null or empty commit_id to 'owner'
                update_query = f"""
                    UPDATE {table_name}
                    SET commit_id = 'MANUAL'
                    WHERE commit_id IS NULL OR commit_id = ''
                """
                result = await conn.execute(update_query)
                print(f"Updated null/empty commit_id in table {table_name}: {result}")
            else:
                print(f"Column 'commit_id' does not exist in table: {table_name}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if conn:
            await conn.close()

# Run the script
asyncio.run(update_null_roles())