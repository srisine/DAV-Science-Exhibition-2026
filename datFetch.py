import ollama

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
        prompt = f"Provide a safe and intended way to dispose a {clsName}. Keep it concise, around 3-4 sentences."
        response = ollama.chat(
            model='llama3.1:8b',
            messages=[{'role':'user', 'content':prompt}]
        )
        return response['message']['content'].strip()
    except Exception:
        return f'Could not connect to ollama rn'
