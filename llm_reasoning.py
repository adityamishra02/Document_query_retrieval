import google.generativeai as genai
import os

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))  # Make sure this is set

model = genai.GenerativeModel("gemini-1.5-flash")  # Use latest stable available

def ask_questions(context: str, questions: list[str]) -> list[str]:
    answers = []

    for question in questions:
        prompt = f"""You are an insurance assistant. Answer based ONLY on the context below:

Context:
\"\"\"
{context}
\"\"\"

Question: {question}
Answer:"""

        response = model.generate_content(prompt)
        answers.append(response.text.strip())

    return answers
