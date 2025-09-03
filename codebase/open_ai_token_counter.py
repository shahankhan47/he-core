import os
import tiktoken



async def open_ai_truncator(text: str, model: str, max_tokens: int):
    try:
        # Initialize the tokenizer
        enc = tiktoken.get_encoding("o200k_base")
        
        # Encode the text to tokens
        tokens = enc.encode(text)
        
        # Check if the number of tokens is within the limit
        if len(tokens) <= max_tokens:
            print(f'length of untruncated file: {len(tokens)}')
            return text
        
        # If too long, truncate and decode the tokens
        else:
            print('file truncated')
            return enc.decode(tokens[:max_tokens])
        
    except Exception as e:
        # Log the exception details
        print(f"An error occurred during truncation: {str(e)}")
        return None
