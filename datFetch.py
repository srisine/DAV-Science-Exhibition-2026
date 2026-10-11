import ollama


def getSort(clsName):
    try:
        categories = [
            "recyclable",
            "e waste",
            "hazardous",
            "organic",
            "generic waste",
            "special disposal",
            "unknown"
        ]

        response = ollama.chat(
            model="llama3.1:8b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a waste classification engine, "
                        "not a conversational assistant. "
                        "Never greet the user, ask questions, "
                        "offer to help, or request an object. "
                        "Classify the given object label only. "
                        'Return JSON exactly like {"category":"recyclable"}. '
                        "Choose only from the allowed categories."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Classify this detected object label: {clsName}. "
                        f"Allowed categories: {categories}. "
                        "Return only the JSON object."
                    )
                }
            ],
            format={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": categories
                    }
                },
                "required": ["category"]
            },
            options={"temperature": 0}
        )

        import json
        result = json.loads(response["message"]["content"])
        category = result["category"]

        if category not in categories:
            return "unknown"

        return category

    except Exception as e:
        print(f"Sorting error: {e}")
        return "unknown"

def getRecycleInfo(clsName):
    try:
        prompt = f"Provide a creative art and craft upcycling idea using a discarded {clsName}. Keep it concise, around 3 to 4 sentences."
        response = ollama.chat(
            model='llama3.1:8b',
            messages=[{'role':'user', 'content':prompt}]
        )
        return response['message']['content'].strip()
    except Exception:
        return f'Could not connect to ollama rn'

def getDisposeInfo(clsName):
    try:
        prompt = f"Provide the safe and intended way to dispose of a {clsName}. Keep it strictly between 3 to 4 sentences. Focus on safety hazards (e.g., sharp edges, e-waste, toxic materials) and proper disposal channels. Use a clear, direct tone and do not include any introductory or concluding text."
        response = ollama.chat(
            model='llama3.1:8b',
            messages=[{'role':'user', 'content':prompt}]
        )
        return response['message']['content'].strip()
    except Exception:
        return f'Could not connect to ollama rn'

def getFacrInfo(clsName):
    try:
        prompt = f"Give a fun, gamified environmental impact fact for recycling a {clsName}. Keep it strictly between 3 to 4 sentences. Use a lively, encouraging tone with score-like framing (e.g., XP, eco-points, or carbon saved). Do not include any extra introduction or concluding text."
        response = ollama.chat(
            model='llama3.1:8b',
            messages=[{'role':'user', 'content':prompt}]
        )
        return response['message']['content'].strip()
    except Exception:
        return f'Could not connect to ollama rn'