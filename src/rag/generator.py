import streamlit as st

from .config import *

if USE_GEMINI:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

class Generator:
    def generate(self, query, docs):
        context = "\n".join(docs)

        if USE_GEMINI:
            prompt = f"""
Question:
{query}

Papers:
{context}

Give a concise and accurate answer.
"""

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are a helpful scientific assistant. Answer using the provided papers."
                )
            )
            
            return response.text

        else:
            return context