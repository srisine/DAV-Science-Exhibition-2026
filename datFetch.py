import json

import ollama


MODEL = "llama3.1:8b"
CATEGORIES = [
    "recyclable",
    "e waste",
    "hazardous",
    "organic",
    "generic waste",
    "special disposal",
    "unknown",
]


def getSort(clsName):
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the supplied detected object label. "
                        "Return only JSON with one key named category. "
                        "Use only an allowed category."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Object: {clsName}. Allowed categories: {CATEGORIES}",
                },
            ],
            format={
                "type": "object",
                "properties": {"category": {"type": "string", "enum": CATEGORIES}},
                "required": ["category"],
            },
            options={"temperature": 0},
        )
        category = json.loads(response["message"]["content"]).get("category", "unknown")
        return category if category in CATEGORIES else "unknown"
    except Exception as exc:
        print(f"Sorting error: {exc}")
        return "unknown"


def _ask(prompt, fallback):
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()
    except Exception:
        return fallback


def getRecycleInfo(clsName):
    return _ask(
        f"Provide a creative art and craft upcycling idea using a discarded {clsName}. Keep it concise, around 3 to 4 sentences.",
        "Could not connect to Ollama right now.",
    )


def getDisposeInfo(clsName):
    return _ask(
        f"Provide the safe and intended way to dispose of a {clsName}. Use 3 to 4 sentences. Focus on hazards and proper disposal channels. Be clear and direct.",
        "Could not connect to Ollama right now.",
    )


def getFacrInfo(clsName):
    return _ask(
        f"Give a fun, gamified environmental impact fact for recycling a {clsName}. Use 3 to 4 lively sentences with eco-points or XP framing.",
        "Could not connect to Ollama right now.",
    )
