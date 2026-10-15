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

    def extract_conversational(self, user_input: str, recent_dialogue: list, cumulative_preferences: dict) -> dict:
        """
        Multi-turn context-aware extractor:
        - Merges delta preferences with existing session state.
        - Resolves overrides (e.g. switching country or place type).
        - Detects dissatisfaction, repetition, and strict conjunction demands ("both mountain and hiking").
        """
        from prompts import CONVERSATION_EXTRACTOR_SYSTEM_PROMPT, build_conversation_extractor_user_prompt

        # 1. Check if LLM is active
        if self.llm is not None and getattr(self.llm, "llm", None) is not None:
            try:
                print(f"[PreferenceExtractor] Running Multi-Turn LLM Extraction on: '{user_input}'...", flush=True)
                user_prompt = build_conversation_extractor_user_prompt(user_input, recent_dialogue, cumulative_preferences)
                response = self.llm.call_llm(CONVERSATION_EXTRACTOR_SYSTEM_PROMPT, user_prompt)
                raw_output = response["choices"][0]["message"]["content"]
                parsed = json.loads(raw_output)
                updated_raw = parsed.get("updated_preferences", {})
                validated_prefs = self.validate_and_format_schema(updated_raw)
                validated_prefs["raw_user_input"] = user_input
                return {
                    "updated_preferences": validated_prefs,
                    "is_dissatisfied": bool(parsed.get("is_dissatisfied", False)),
                    "strict_conjunction": bool(parsed.get("strict_conjunction", False)),
                    "wants_fresh_alternatives": bool(parsed.get("wants_fresh_alternatives", False)),
                    "intent_summary": parsed.get("user_intent_summary", "Refining travel preferences.")
                }
            except Exception as e:
                print(f"[PreferenceExtractor] Multi-turn LLM notice: {e}. Falling back to rule-based context merger.", flush=True)

        # 2. Heuristic Context Merger (Fallback)
        text_lower = user_input.lower()
        delta_prefs = self.fallback_extract(user_input)

        # Clone existing cumulative state
        merged = {
            "duration": cumulative_preferences.get("duration", "unspecified"),
            "place_of_interests": list(cumulative_preferences.get("place_of_interests", [])),
            "vibe": list(cumulative_preferences.get("vibe", [])),
            "place_of_activites": list(cumulative_preferences.get("place_of_activites", [])),
            "specific_named_places": list(cumulative_preferences.get("specific_named_places", [])),
            "country_names": list(cumulative_preferences.get("country_names", [])),
            "city_names": list(cumulative_preferences.get("city_names", [])),
            "raw_user_input": user_input
        }

        # Override Detection: "instead", "change to", "switch to", "what if we went to"
        is_override = any(w in text_lower for w in ["instead", "change to", "switch to", "not in", "what about", "what if we went to"])
        if is_override:
            if delta_prefs["country_names"]:
                merged["country_names"] = delta_prefs["country_names"]
            if delta_prefs["city_names"]:
                merged["city_names"] = delta_prefs["city_names"]
            if delta_prefs["place_of_interests"]:
                merged["place_of_interests"] = delta_prefs["place_of_interests"]
        else:
            # Additive merge
            for key in ["country_names", "city_names", "place_of_interests", "vibe", "place_of_activites", "specific_named_places"]:
                for item in delta_prefs.get(key, []):
                    if item and item not in merged[key]:
                        merged[key].append(item)

        if delta_prefs.get("duration") != "unspecified":
            merged["duration"] = delta_prefs["duration"]

        # Dissatisfaction Detection
        dissatisfaction_triggers = [
            "don't like", "dont like", "other places", "different", "already been",
            "something else", "not satisfied", "change these", "none of these",
            "show me other", "show other", "give me other", "too crowded", "not what i want",
            "does not match", "doesnt match", "not match", "other than this", "other than these",
            "something different other than", "wrong places"
        ]
        is_dissatisfied = any(trig in text_lower for trig in dissatisfaction_triggers)

        # Strict Conjunction / Spatial Proximity Detection: "both ... and", "must have", "near", "together"
        conjunction_triggers = [
            "both", "all of these", "in that same", "together", "in that mountain",
            "not either", "must have both", "want both", "must have a beach", "must have",
            "near the place", "near the", "nearby", "close to the beach", "by the sea",
            "coastal mountain", "near the ocean", "combined with"
        ]
        strict_conjunction = any(trig in text_lower for trig in conjunction_triggers)

        wants_fresh_alternatives = is_dissatisfied or any(w in text_lower for w in ["more", "other options", "next batch", "fresh", "different other than"])

        return {
            "updated_preferences": merged,
            "is_dissatisfied": is_dissatisfied,
            "strict_conjunction": strict_conjunction,
            "wants_fresh_alternatives": wants_fresh_alternatives,
            "intent_summary": "Refined preferences based on conversation context."
        }


if __name__ == "__main__":
    extractor = PreferenceExtractor()
    sample_text = "I want to go on a 1 year break in Australia with mountain adventure, hiking in Blue Mountains."
    extractor.extract(sample_text)

