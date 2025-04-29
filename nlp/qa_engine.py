import json
import random
from difflib import get_close_matches

class KinyarwandaQA:
    def __init__(self, intents_file="nlp/intents.json"):
        with open(intents_file) as f:
            self.intents = json.load(f)

    def get_response(self, query, threshold=0.6):
        query = query.lower().strip()
        for intent, data in self.intents.items():
            for pattern in data["patterns"]:
                if pattern in query:
                    return random.choice(data["responses"])
        matches = get_close_matches(query, self.intents.keys(), n=1, cutoff=threshold)
        if matches:
            return self.intents[matches[0]]["responses"][0]
        return "Sinzi igisubizo cy'icyo kibazo."