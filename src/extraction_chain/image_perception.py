
import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

REASONING_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(REASONING_API_KEY)

def chat_completion(prompt, model="gpt-4o", role="user"):


    messages = [{"role": role, "content": prompt}]

    response = client.chat.completions.create(model=model,
    messages=messages)

    output = response.choices[0].message.content
    return output

