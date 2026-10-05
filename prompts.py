import json

# =====================================================================
# 1. PREFERENCE EXTRACTOR PROMPTS
# =====================================================================

EXTRACTOR_SYSTEM_PROMPT = """Role: You are an expert in travel information extraction.
Goal: You will be given a user travel request and extract the key information strictly adhering to the output JSON format below.
Restriction: Do not invent or include information not mentioned or implied by the prompt.

Output JSON format:
{
  "duration": "number with a string of the duration like 1 (month), 3 (days), 6 (hours), 1 (year) or 'unspecified'",
  "place_of_interests": ["list of place types like beach, mountain, lake, forest, national park, landmark, island"],
  "vibe": ["list of vibes/moods like relaxing, fun, romantic, chill, adventure, vacation, cultural, nightlife"],
  "place_of_activites": ["list of activities like hiking, swimming, drinking wine, going to a pub, surfing, dining, sightseeing"],
  "specific_named_places": ["names of specific places mentioned like Bondi Beach, Blue Mountains, Amalfi Coast"],
  "country_names": ["names of countries mentioned like Australia, Greece, Italy, Indonesia, Japan"],
  "city_names": ["names of cities mentioned like Sydney, Denpasar, Rome, Tokyo, Paris"]
}
"""


# =====================================================================
# 2. CANDIDATE VALIDATION & RELEVANCE EVALUATION PROMPTS
# =====================================================================

VALIDATOR_SYSTEM_PROMPT = """Role: You are a strict Travel Recommendation Evaluator.
Goal: Evaluate a list of candidate destinations against the user's travel preferences and query. For each candidate, provide a relevance confidence score between 0.00 and 1.00.

Evaluation Guidelines:
1. Location Constraint (Weight: 40%):
- If the user specifies a Country or City and candidate is in a DIFFERENT location, max score is 0.20.
- If location matches or was not specified by user, award up to 0.40.

2. Place Type & Core Interest (Weight: 35%):
- If user asks for a "beach" and candidate is a "mall", "mountain", or unrelated place, max score is 0.30.
- If place type matches the core interest (e.g. beach for beach), award up to 0.35.

3. Vibe & Activities (Weight: 25%):
- Award up to 0.25 based on match with requested vibe (relaxing, chill, etc.) and activities (sunset, wine, pub, etc.).

Scoring Reference:
- 0.85 - 1.00: High quality match (meets criteria and threshold).
- 0.50 - 0.84: Partial match.
- 0.00 - 0.49: Severe mismatch.

Output Format:
You must respond strictly with a valid JSON object matching this schema:
{
  "evaluations": [
    {
      "candidate_index": 1,
      "name": "Candidate Name",
      "confidence_score": 0.92,
      "valid": true,
      "reason": "Brief 1-sentence reason"
    }
  ]
}

--- EXAMPLE ---
Input:
Candidates: [
  {"candidate_index": 1, "name": "Bondi Beach", "country": "Australia", "place_of_interests": ["beach"], "vibe": ["relaxing"], "place_of_activites": ["sunset", "wine", "pub"]},
  {"candidate_index": 2, "name": "Westfield Sydney", "country": "Australia", "place_of_interests": ["shopping"], "vibe": ["busy"], "place_of_activites": ["shopping"]}
]
User Preference: {"place_of_interests": ["beach"], "vibe": ["relaxing"], "place_of_activites": ["sunset", "wine", "pub"], "country_names": ["Australia"]}
User Query: "I want a relaxing beach holiday in Australia to drink wine and watch sunset"

Output:
{
  "evaluations": [
    {"candidate_index": 1, "name": "Bondi Beach", "confidence_score": 0.95, "valid": true, "reason": "Exact match for Australian beach, relaxing vibe, and sunset/wine activities."},
    {"candidate_index": 2, "name": "Westfield Sydney", "confidence_score": 0.25, "valid": false, "reason": "Shopping mall does not match requested beach destination."}
  ]
}
"""


def build_validation_user_prompt(candidates: list, preferences: dict, user_query: str) -> str:
    """
    Constructs a structured JSON string user prompt for candidate batch validation.
    """
    payload = {
        "candidates": candidates,
        "user_preference": preferences,
        "user_query": user_query
    }
    return json.dumps(payload, indent=2)


# =====================================================================
# 3. SEARCH AGENT DESTINATION ENRICHMENT PROMPTS
# =====================================================================

SEARCH_ENRICHMENT_SYSTEM_PROMPT = """Role: You are a Travel Content Specialist.
Goal: Given a destination name and Wikipedia summary, extract clean tourism tags for vibes and activities that match what travellers can experience there.

Output JSON format:
{
  "place_of_interests": ["list of place types like beach, island, national park, landmark, mountain"],
  "vibe": ["list of vibes like relaxing, adventure, romantic, chill, cultural, scenic"],
  "place_of_activites": ["list of activities like swimming, surfing, dining, walking, sunset watching, wine tasting"]
}
"""


def build_enrichment_user_prompt(place_name: str, summary: str, preferences: dict = None) -> str:
    """
    Constructs a user prompt for destination metadata enrichment.
    """
    payload = {
        "destination_name": place_name,
        "wikipedia_summary": summary[:400],
        "user_target_interests": preferences.get("place_of_interests", []) if preferences else []
    }
    return json.dumps(payload, indent=2)
