import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

from dotenv import load_dotenv
load_dotenv()


class LLMAgent:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment. Please check your .env file.")
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    async def answer(self, question: str, context: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful shop assistant. Use the provided context to answer the user naturally."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content

llm_agent = None

def get_llm_agent():
    global llm_agent
    if llm_agent is None:
        llm_agent = LLMAgent()
    return llm_agent