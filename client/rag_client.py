import requests

API_URL = "http://localhost:8000/rag"

def query_rag(prompt):
    continuation_token = None
    full_answer = ""

    while True:
        payload = {
            "prompt": prompt if continuation_token is None else None,
            "continuation_token": continuation_token,
            "max_tokens": 256,
            "model": "llama3.1:8b"
        }
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()

        full_answer += data["text"]

        if not data["continue"]:
            break
        continuation_token = data["continuation_token"]

    return full_answer

if __name__ == "__main__":
    prompt = input("Enter your question: ")
    answer = query_rag(prompt)
    print("\nAnswer:\n", answer)
