import os
import anthropic
import json
from typing import Dict, Any
client = anthropic.Anthropic()
client.api_key = os.getenv('ANTHROPIC_API_KEY')


async def generate_diagrams( project_id: str,  user_question: str, summary: str) -> str:

    try:

        api_messages = []
        api_messages.append({"role": "user", "content": user_question})
    
        mermaid_tool = [
                         {
                             "name" : "mermaid_diagram_generator",
                             "description"  : "generates a mermaid diagram based on the mermaid code provided",
                             "input_schema"  : {
                                                  "type" : "object", 
                                                   "properties": { 
                                                    "mermaid_code" : {
                                                                        "type" : "string",
                                                                        "description" : "The code will be run in a mermaid renderer"
                                                                      }
                                                                 },

                                 "required" : ["ticker"]
                                     
                                 }
                            }
                        
                         ] 
        # Print api_messages for debugging
        print("API Messages:", api_messages)

        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=2000,
            temperature = 0.0,
            tool_choice = {"type": "tool", "name": "mermaid_diagram_generator"},
            tools = mermaid_tool,
            system=[
                    {"type": "text", "text": "You are an AI Production owner  tasked with analyzing codebases for Technicsl, User Journerys, Features or over all flow. Based on the users request you must provide a Mermid diagram only."},
                    {"type": "text", "text": f"Here is the summary of a complex codebase: {summary}", "cache_control": {"type": "ephemeral"}}
                ],
            messages=api_messages
        )

        # Add the AI's response to the conversation history
        try:
           if response.stop_reason == 'tool_use':
               mermaid_code = response.content[0].input
           else:
               raise Exception(" Error calling the mermaid generation core api")
               
        except Exception as e:
            return str(e) 
        
        return mermaid_code
    except Exception as e:
        return str(e)




























