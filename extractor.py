import json
import re
from abstract_llm import AbstractLLM
from prompts import EXTRACTOR_SYSTEM_PROMPT

DEFAULT_SCHEMA = {
    "duration": "unspecified",
    "place_of_interests": [],
    "vibe": [],
    "place_of_activites": [],
    "specific_named_places": [],
    "country_names": [],
    "city_names": []
}

class PreferenceExtractor:
    """
    Preference Extractor:
    - LLM-driven information extraction using LlamaCPP (gemma-4-E2B-it-GGUF).
    - Includes dynamic text parsing fallback so key preferences (e.g. beach, relaxing, sunset, pub) are never lost even if LLM is offline.
    """
    def __init__(self, llm: AbstractLLM = None):
        if llm is not None:
            self.llm = llm
        else:
            try:
                print("[PreferenceExtractor] Loading LlamaCPP LLM (ggml-org/gemma-4-E2B-it-GGUF)...", flush=True)
                self.llm = AbstractLLM() 
                print("[PreferenceExtractor] LlamaCPP LLM loaded successfully!", flush=True)
            except Exception as e:
                print(f"[PreferenceExtractor] LlamaCPP model initialization notice: {e}", flush=True)
                self.llm = None

    def validate_and_format_schema(self, raw_data: dict) -> dict:
        """
        Validates and formats the extracted LLM JSON to guarantee exact schema alignment.
        """
        formatted = {}
        
        # Duration key
        duration_val = raw_data.get("duration", "unspecified")
        formatted["duration"] = str(duration_val) if duration_val else "unspecified"
        
        # List keys
        list_keys = [
            "place_of_interests",
            "vibe",
            "place_of_activites",
            "specific_named_places",
            "country_names",
            "city_names"
        ]
        
        for key in list_keys:
            val = raw_data.get(key, [])
            if isinstance(val, list):
                formatted[key] = [str(item) for item in val if item]
            elif isinstance(val, str) and val:
                formatted[key] = [val]
            else:
                formatted[key] = []

        return formatted

    def fallback_extract(self, user_input: str) -> dict:
        """
        Extracts key preferences dynamically from raw text when LLM is offline.
        """
        text = user_input.lower()
        
        interests = []
        for word in ["beach", "mountain", "ocean", "lake", "forest", "scenery", "views", "shopping", "food", "park", "museum", "island", "trail"]:
            if word in text:
                interests.append(word)

        vibes = []
        for word in ["relaxing", "vacation", "adventure", "nature", "romantic", "chill", "fun", "city", "foodie", "warm", "sunny", "tropical", "scenic"]:
            if word in text:
                vibes.append(word)

        activities = []
        for phrase in ["sunset", "wine", "pub", "bar", "restaurant", "hiking", "camping", "camp", "swimming", "fishing", "surfing", "shopping", "sightseeing", "walking"]:
            if phrase in text:
                activities.append(phrase)

        countries = []
        for c in ["Australia", "Japan", "USA", "France", "Italy", "China", "Denmark"]:
            if c.lower() in text:
                countries.append(c)

        cities = []
        for ct in ["Sydney", "Tokyo", "Paris", "London", "Rome", "Bondi", "Lithgow"]:
            if ct.lower() in text:
                cities.append(ct)

        result = {
            "duration": "unspecified",
            "place_of_interests": interests,
            "vibe": vibes,
            "place_of_activites": activities,
            "specific_named_places": [],
            "country_names": countries,
            "city_names": cities,
            "raw_user_input": user_input
        }
        return result

    def extract(self, user_input: str) -> dict:
        """
        Extracts preferences dynamically using the LLM and formats key-value pairs.
        """
        if self.llm is not None:
            try:
                print(f"[PreferenceExtractor] Running LLM extraction on prompt: '{user_input}'...", flush=True)
                response = self.llm.call_llm(EXTRACTOR_SYSTEM_PROMPT, user_input)
                raw_output = response["choices"][0]["message"]["content"]
                parsed = json.loads(raw_output)
                validated = self.validate_and_format_schema(parsed)
                validated["raw_user_input"] = user_input
                print(f"[PreferenceExtractor] Extracted Preferences:\n{json.dumps(validated, indent=2)}", flush=True)
                return validated
            except Exception as e:
                print(f"[PreferenceExtractor] LLM Extraction Error: {e}", flush=True)

        print("[PreferenceExtractor] LLM not active. Using dynamic keyword fallback.", flush=True)
        fallback_res = self.fallback_extract(user_input)
        print(f"[PreferenceExtractor] Fallback Preferences:\n{json.dumps(fallback_res, indent=2)}", flush=True)
        return fallback_res

if __name__ == "__main__":
    extractor = PreferenceExtractor()
    sample_text = "I want to go on a 1 year break in Australia with mountain adventure, hiking in Blue Mountains."
    extractor.extract(sample_text)
