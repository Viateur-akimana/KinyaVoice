import json
import random
from difflib import get_close_matches

class KinyarwandaQA:
    def __init__(self, intents_file="nlp/intents.json"):
        with open(intents_file) as f:
            self.intents = json.load(f)
            
    def get_response(self, query, threshold=0.6):
        """
        Get a response for the given query by matching it with patterns in intents.
        
        Args:
            query (str): The user's query/input.
            threshold (float): Minimum similarity threshold for fuzzy matching.
            
        Returns:
            str: The response to the query.
        """
        query = query.lower().strip()
        
        # First, try direct pattern matching within intents
        for intent, data in self.intents.items():
            for pattern in data["patterns"]:
                if pattern.lower() in query:
                    return random.choice(data["responses"])
        
        # If no direct match, try fuzzy matching the entire query
        # against patterns in all intents
        all_patterns = []
        pattern_to_intent = {}
        
        for intent, data in self.intents.items():
            for pattern in data["patterns"]:
                all_patterns.append(pattern.lower())
                pattern_to_intent[pattern.lower()] = intent
        
        matches = get_close_matches(query, all_patterns, n=1, cutoff=threshold)
        if matches:
            matched_intent = pattern_to_intent[matches[0]]
            return random.choice(self.intents[matched_intent]["responses"])
        
        # If still no match, return default response
        return "Saa mbiri z'umugoroba"  # "I don't know the answer to that question"