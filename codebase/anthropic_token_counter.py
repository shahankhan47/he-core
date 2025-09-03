import anthropic
import os

api_key = os.getenv('ANTHROPIC_API_KEY')
client = anthropic.Anthropic(api_key=api_key)




def anthropic_truncator(text, max_tokens=160000, model="claude-3-5-sonnet-20241022"):
    """Use binary search to efficiently truncate text to fit token limit."""
    
    # Initial check
    count = client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}]
    )
    
    if count.input_tokens <= max_tokens:
        return text
    
    # Binary search for the right length
    low = 0
    high = len(text)
    
    while high - low > 5000:  # Stop when we're close enough
        mid = (low + high) // 2
        truncated = text[:mid]
        
        count = client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": truncated}]
        )
        
        if count.input_tokens <= max_tokens:
            low = mid
        else:
            high = mid
    
    # Fine-tune the final result
    while True:
        truncated = text[:high]
        count = client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": truncated}]
        )
        
        if count.input_tokens <= max_tokens:
            return truncated
        
        high -= 5000  # Reduce by 5000 characters at a time until we fit