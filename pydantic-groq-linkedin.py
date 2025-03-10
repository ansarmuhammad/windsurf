from pydantic import BaseModel
from groq import Groq

class GroqAPIRequest(BaseModel):
    api_key: str
    prompt: str = "You are a wise assistant. Provide profound, meaningful phrases of wisdom. Each phrase should be 50-120 characters, contain deep insights, and use sophisticated language structure.\n\nFocus on universal truths and philosophical insights.\n\nSeparate each phrase with a newline."
    max_phrases: int = 5
    model: str = "mixtral-8x7b-32768"

def get_phrases_of_wisdom(request_data: GroqAPIRequest):
    try:
        # Initialize Groq client
        client = Groq(api_key=request_data.api_key)
        
        # Generate completion
        completion = client.chat.completions.create(
            model=request_data.model,
            messages=[
                {"role": "system", "content": "You are a wise assistant. Provide concise, meaningful phrases of wisdom."},
                {"role": "user", "content": request_data.prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        # Extract phrases from response
        if completion.choices and completion.choices[0].message.content:
            phrases = completion.choices[0].message.content.strip().split('\n')
            return phrases[:request_data.max_phrases]
        return []
    except Exception as e:
        print(f"Request failed: {e}")
        return [] 
if __name__ == "__main__":
    # Replace with your actual Groq API key
    request_data = GroqAPIRequest(api_key="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")
    phrases = get_phrases_of_wisdom(request_data)
    for i, phrase in enumerate(phrases, 1):
        print(f"{i}. {phrase}")
