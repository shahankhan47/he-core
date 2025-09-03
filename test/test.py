import os
import openai
import json
import requests
import csv

# Set your OpenAI API Key
openai.api_key = os.environ.get("OPENAI_API_KEY") 

#prod
"""save_file_name = "harmonyprod_12525.csv"
Harmony_Engine_URL = "https://harmonyenginev3-dev-production.up.railway.app/"
project_id = "c129d2cd-4d93-45f9-b3ab-cfe41aa2b23a"
email = "saiganesh977@gmail.com"
api_key = os.environ["API_KEY"]  # Harmony engine key"""


#dev
save_file_name = "harmonydev_12525new.csv"
Harmony_Engine_URL = "https://harmonyenginev3-dev-dev.up.railway.app/"
project_id = "d13efe35-75ac-49d0-ab92-9496943cfa2c"
email = "saiganesh977@gmail.com"
api_key = os.environ["API_KEY"]  



def chat_with_codebase(api_key, url, project_id, email, user_question) -> str:
    endpoint = f"{url}/chat-pro"
    headers = {"x-api-key":api_key}
    form_data = {"user_question": user_question, "project_id": project_id, "email": email}
    try:
        response = requests.post(endpoint, headers=headers, data=form_data)
        response.raise_for_status()
        return response.json()["result"]
    except Exception as e:
        return f"ERROR: {e}"

def judge_llm(base_answer: str, llm_answer: str, question: str, openai_model="gpt-4.1"):
    prompt = f"""
You are to act as an impartial expert and evaluate an LLM-generated answer.
Here are the evaluation criteria, each to be rated as a number from 0 to 10 (integer):

1. Correctness or accuracy of the information: 0=totally incorrect, 10=absolutely correct.
2. Amount of possible hallucination (made up or fabricated or irrelevant content): 0=none, 10=entirely hallucinated.
3. Crispness of the information (how succinct, concise, and straight-to-the-point is the answer): 0=verbose, rambling, lengthy for no reason; 10=super crisp/concise.
4. Completeness (does it fully answer the user request?): 0=completely missing, 10=fully complete.

Instructions:
- Compare the LLM-generated answer to the BASE ANSWER for the given QUESTION.
- Give each criterion a score from 0 to 10 (higher is better except hallucination; for it, 0 is best).
- Write a one-line observation.

Respond with only a valid JSON object, like:
{{"correctness": <int>, "hallucination": <int>, "crispness": <int>, "completeness": <int>, "observation": "<one line>"}}

QUESTION:
{question}

BASE ANSWER:
{base_answer}

LLM-GENERATED ANSWER:
{llm_answer}
"""
    completion = openai.chat.completions.create(
        model=openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=600
    )
    content = completion.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except Exception:
        print(f"OpenAI judge output could not be parsed as JSON:\n{content}\n")
        return {
            "correctness": "",
            "hallucination": "",
            "crispness": "",
            "completeness": "",
            "observation": f"Could not parse: {content}"
        }

def main():
    with open('./test/test_data.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    rows = []
    for entry in test_data["Questions"]:
        question_id = entry["Id"]
        question = entry["Question"]
        base_answer = entry["Answer"]

        print(f"\nQUESTION [{question_id}]: {question}")

        # Get LLM answer from API
        llm_answer = chat_with_codebase(
            api_key=api_key,
            url=Harmony_Engine_URL,
            project_id=project_id,
            email=email,
            user_question=question
        )
        print(f"LLM ANSWER:\n{llm_answer}\n")

        judge_scores = judge_llm(base_answer, llm_answer, question)
        print(f"JUDGE OUTPUT:\n{judge_scores}\n")

        # Prepare CSV row
        rows.append([
            question_id,
            question,
            llm_answer,
            base_answer,
            judge_scores.get("correctness", ""),
            judge_scores.get("hallucination", ""),
            judge_scores.get("crispness", ""),
            judge_scores.get("completeness", ""),
            judge_scores.get("observation", ""),
        ])

    # Save as CSV file
    with open(save_file_name, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "question", "llm_answer", "base_answer",
            "correctness", "hallucination", "crispness", "completeness", "observation"
        ])
        writer.writerows(rows)
    print("Judging complete. Results saved to judge_results.csv\n")

if __name__ == "__main__":
    main()