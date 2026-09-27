import sys
import json
from extractor import PreferenceExtractor
from database_agent import DatabaseAgent
from search_agent import SearchAgent
from abstract_llm import AbstractLLM


class OrchestratorAgent:
    """
    Orchestrator Agent:
    - Chats with users about desired tourism preferences.
    - Extracts structured requirements.
    - Calls Database Agent (Chroma DB via LlamaIndex).
    - Calls Search Agent (web search fallback/enrichment).
    - Evaluates batch candidates in a single LLM call with strict >= 85% confidence.
    """

    def __init__(self):
        self.SYSTEM_PROMPT = """Role: You are a strict Travel Recommendation Evaluator.
Goal: Evaluate a list of candidate destinations against the user's travel preferences and query. For each candidate, provide a relevance confidence score between 0.00 and 1.00.

Evaluation Guidelines:
1. Location Constraint (Weight: 40%):
- If the user specifies a Country or City and candidate is in a DIFFERENT location, max score is 0.20.
- If location matches or was not specified by user, award up to 0.40.

2. Place Type & Core Interest (Weight: 35%):
- If user asks for a "beach" and candidate is a "mall", "mountain", or unrelated, max score is 0.30.
- If place type matches the core interest, award up to 0.35.

3. Vibe & Activities (Weight: 25%):
- Award up to 0.25 based on match with requested vibe (relaxing, chill, etc.) and activities (sunset, wine, pub, etc.).

Scoring Reference:
- 0.85 - 1.00: High quality match.
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
        print("[Orchestrator] Initializing Multi-Agent System...")
        self.llm = AbstractLLM()
        self.extractor = PreferenceExtractor(llm=self.llm)
        self.db_agent = DatabaseAgent()
        self.search_agent = SearchAgent()
        print("[Orchestrator] Ready to serve tourism requests!")

    def check_validity(self, place_results: list, preferences: dict, user_input: str, min_confidence: float = 0.85) -> tuple[bool, list]:
        """
        Check validity & confidence for candidate places in a single batch LLM call.
        Enforces confidence >= min_confidence (85%).
        """
        if not place_results:
            return False, []

        print(f"\n[Orchestrator] Performing Batch Evaluation on {len(place_results)} candidate(s)...", flush=True)

        candidates_payload = [
            {
                "candidate_index": idx,
                "name": p.get("name", ""),
                "country": p.get("country", ""),
                "place_of_interests": p.get("place_of_interests", []),
                "vibe": p.get("vibe", []),
                "place_of_activites": p.get("place_of_activites", []),
                "summary": str(p.get("text_content", ""))[:200]
            }
            for idx, p in enumerate(place_results, 1)
        ]

        USER_PROMPT = json.dumps({
            "candidates": candidates_payload,
            "user_preference": preferences,
            "user_query": user_input
        }, indent=2)

        evaluations = []
        if self.llm and self.llm.llm is not None:
            try:
                response = self.llm.call_llm(self.SYSTEM_PROMPT, USER_PROMPT)
                raw_content = response["choices"][0]["message"]["content"]
                parsed = json.loads(raw_content)
                evaluations = parsed.get("evaluations", []) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])
            except Exception as e:
                print(f"[Orchestrator] Batch LLM evaluation error: {e}. Falling back to heuristic scoring.", flush=True)

        eval_map = {}
        for ev in evaluations:
            if "candidate_index" in ev:
                eval_map[ev["candidate_index"]] = ev
            if "name" in ev:
                eval_map[str(ev["name"]).strip().lower()] = ev

        valid_places = []
        for idx, place in enumerate(place_results, 1):
            name_key = str(place.get("name", "")).strip().lower()
            ev = eval_map.get(idx) or eval_map.get(name_key)

            if ev and "confidence_score" in ev:
                try:
                    score = float(ev["confidence_score"])
                except (ValueError, TypeError):
                    score = self.search_agent.evaluate_confidence(place.get("name", ""), place.get("text_content", ""), preferences)
                reason = ev.get("reason", "LLM batch evaluated.")
            else:
                score = self.search_agent.evaluate_confidence(place.get("name", ""), place.get("text_content", ""), preferences)
                reason = "Heuristic match."

            score = min(1.0, max(0.0, round(score, 2)))
            place["confidence_score"] = score
            place["confidence"] = f"{int(score * 100)}%"
            place["evaluation_reason"] = reason

            title = place.get("name", "Unknown")
            if score >= min_confidence:
                print(f"[Orchestrator] Candidate #{idx} '{title}' -> Confidence: {place['confidence']} (PASSED >= {int(min_confidence * 100)}%) | Reason: {reason}", flush=True)
                valid_places.append(place)
            else:
                print(f"[Orchestrator] Candidate #{idx} '{title}' -> Confidence: {place['confidence']} (FAILED < {int(min_confidence * 100)}%) | Reason: {reason}", flush=True)

        if valid_places:
            print(f"[Orchestrator] Validity check PASSED: {len(valid_places)} matched with >= {int(min_confidence * 100)}% confidence.\n", flush=True)
            return True, valid_places
        else:
            print(f"[Orchestrator] Validity check FAILED: No candidate passed >= {int(min_confidence * 100)}% confidence threshold.\n", flush=True)
            return False, []

    def process_request(self, user_input: str) -> str:
        """
        Full orchestration workflow:
        1. Extract preferences via PreferenceExtractor
        2. Query ChromaDB via DatabaseAgent
        3. Batch evaluate database candidates (>= 85% confidence)
        4. If database returns are insufficient/invalid, query SearchAgent
        5. Batch evaluate search candidates (>= 85% confidence)
        6. Return formatted recommendation or graceful fallback message
        """
        print(f"\n--- New Request: '{user_input}' ---")

        # Step 1: Extract preferences
        preferences = self.extractor.extract(user_input)

        # Step 2: Call Database Agent (Chroma DB + LlamaIndex)
        db_results = self.db_agent.search_database(preferences)

        # Step 3: Check validity of database results in batch
        is_valid_db, valid_db_places = self.check_validity(db_results, preferences, user_input, min_confidence=0.85)

        if is_valid_db:
            return self.format_response(valid_db_places, source="Local Database (Chroma + LlamaIndex)")

        # Step 4: If database return invalid or empty, call Search Agent
        print("[Orchestrator] Database results insufficient or below 85% confidence threshold. Calling Search Agent...", flush=True)
        search_results = self.search_agent.search_online(preferences)

        # Step 5: Check validity of search results in batch
        is_valid_search, valid_search_places = self.check_validity(search_results, preferences, user_input, min_confidence=0.85)

        if is_valid_search:
            return self.format_response(valid_search_places, source="Web Search Agent")

        # Step 6: If not valid, return apology message
        return "Sorry, I can't work on this request as no destination in our system or online search matched your exact preferences. Please try refining your request!"

    def _clean_ascii(self, text: str) -> str:
        if not text:
            return ""
        return str(text).encode("ascii", errors="replace").decode("ascii").replace("?", "")

    def format_response(self, places: list, source: str) -> str:
        """
        Formats recommendations nicely for user display.
        """
        lines = [f"=== Tourism Recommendations (Source: {source}) ==="]
        for idx, place in enumerate(places, 1):
            name = self._clean_ascii(place.get('name', 'Unknown'))
            lines.append(f"\n[Place #{idx}] {name}")
            if place.get("confidence"):
                lines.append(f"   Match Confidence: {place.get('confidence')}")
            country = self._clean_ascii(place.get('country', 'N/A'))
            state = self._clean_ascii(place.get('state', ''))
            locality = self._clean_ascii(str(place.get('locality', '')))
            lines.append(f"   Country: {country} | State/Locality: {state} {locality}")

            lat = place.get('latitude', 0.0)
            lon = place.get('longitude', 0.0)
            if lat != 0.0 or lon != 0.0:
                lines.append(f"   Coordinates: Lat {lat}, Lon {lon}")

            desc = self._clean_ascii(place.get('text_content', ''))
            lines.append(f"   Description: {desc}")

            if place.get("vibe"):
                lines.append(f"   Vibe: {self._clean_ascii(str(place.get('vibe')))}")
            if place.get("place_of_activites"):
                lines.append(f"   Activities: {self._clean_ascii(str(place.get('place_of_activites')))}")
            if place.get("evaluation_reason"):
                lines.append(f"   Match Details: {self._clean_ascii(str(place.get('evaluation_reason')))}")
            if place.get("image_url"):
                lines.append(f"   Image: {place.get('image_url')}")
            if place.get("source_url"):
                lines.append(f"   Source URL: {place.get('source_url')}")
        return "\n".join(lines)

    def run_interactive_chat(self):
        print("\n=======================================================")
        print(" Welcome to Tourism Search Assistant (Multi-Agent System)")
        print(" Type your dream travel plans or preferences (or 'exit' to quit)")
        print("=======================================================\n")

        while True:
            try:
                user_prompt = input("User > ")
                if not user_prompt.strip():
                    continue
                if user_prompt.lower().strip() in ["exit", "quit", "q"]:
                    print("Goodbye! Safe travels!")
                    break

                response = self.process_request(user_prompt)
                print(f"\nOrchestrator >\n{response}\n")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break


if __name__ == "__main__":
    orchestrator = OrchestratorAgent()

    if len(sys.argv) > 1:
        test_prompt = " ".join(sys.argv[1:])
        res = orchestrator.process_request(test_prompt)
        print("\n" + res)
    else:
        demo_prompt = "I want to go on a relaxing beach where i can see the sunset and drink some wine go to a pub and have a relaxing time"
        print(f"Running Demo Query: '{demo_prompt}'")
        res = orchestrator.process_request(demo_prompt)
        print("\n" + res)
