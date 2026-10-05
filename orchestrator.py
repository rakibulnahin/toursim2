import sys
import json
import os
from extractor import PreferenceExtractor
from database_agent import DatabaseAgent
from search_agent import SearchAgent
from abstract_llm import AbstractLLM
from prompts import VALIDATOR_SYSTEM_PROMPT, build_validation_user_prompt


class OrchestratorAgent:
    """
    Orchestrator Agent (Hybrid Multi-Agent Tourism System):
    - Chats with users about desired tourism preferences.
    - Extracts structured requirements via PreferenceExtractor.
    - Queries Database Agent (Chroma DB via LlamaIndex).
    - Calls Search Agent in Hybrid Mode to return at least 10 high-quality suggestions worldwide.
    - Evaluates batch candidates in a single LLM call using prompts from 'prompts.py'.
    - Automatically saves newly discovered search destinations into ChromaDB for future queries.
    """

    def __init__(self):
        print("[Orchestrator] Initializing Multi-Agent System...")
        # 1. Initialize the single shared LLM instance
        self.llm = AbstractLLM()

        # 2. Inject shared LLM into all agents
        self.extractor = PreferenceExtractor(llm=self.llm)
        self.db_agent = DatabaseAgent()
        self.search_agent = SearchAgent(llm=self.llm)
        print("[Orchestrator] Ready to serve tourism requests!")

    def check_validity(self, place_results: list, preferences: dict, user_input: str, min_confidence: float = 0.75) -> tuple[bool, list]:
        """
        Fast Heuristic Pre-Filtering + Consolidated LLM Validation:
        1. Instantly scores all candidates using heuristic rules (<1ms). Discards severe mismatches (score < 0.40).
        2. Validates only the top promising candidates in a single consolidated LLM call (instead of multiple slow loops).
        3. Enforces confidence >= min_confidence.
        """
        if not place_results:
            return False, []

        print(f"\n[Orchestrator] Performing Fast Heuristic Pre-Filtering on {len(place_results)} candidate(s)...", flush=True)

        # Step 1: Fast Heuristic Pre-Score (Runs in <1ms)
        scored_candidates = []
        for idx, place in enumerate(place_results, 1):
            h_score = self.search_agent.evaluate_confidence(
                place.get("name", ""),
                place.get("text_content", ""),
                preferences
            )
            place["heuristic_score"] = h_score
            # Pre-filter: keep candidates with reasonable potential (heuristic score >= 0.40)
            if h_score >= 0.40:
                scored_candidates.append((idx, place))
            else:
                name_lbl = self._clean_ascii(place.get("name", "Unknown"))
                print(f"[Orchestrator] Candidate #{idx} '{name_lbl}' dropped early by Heuristic Filter (score: {int(h_score*100)}% < 40%)", flush=True)

        if not scored_candidates:
            print(f"[Orchestrator] Validity check: No candidate passed heuristic pre-filtering.", flush=True)
            return False, []

        # Sort promising candidates by heuristic score descending
        scored_candidates.sort(key=lambda x: x[1].get("heuristic_score", 0.0), reverse=True)

        # Step 2: Validate the top candidates in a SINGLE consolidated LLM call (max 8 candidates)
        top_candidates = scored_candidates[:8]
        evaluations = []

        if self.llm and self.llm.llm is not None:
            candidates_payload = [
                {
                    "candidate_index": orig_idx,
                    "name": p.get("name", ""),
                    "country": p.get("country", ""),
                    "place_of_interests": p.get("place_of_interests", []),
                    "vibe": p.get("vibe", []),
                    "place_of_activites": p.get("place_of_activites", []),
                    "summary": str(p.get("text_content", ""))[:140]
                }
                for orig_idx, p in top_candidates
            ]

            user_prompt = build_validation_user_prompt(candidates_payload, preferences, user_input)
            print(f"[Orchestrator] Sending single consolidated batch of {len(candidates_payload)} candidate(s) to LLM...", flush=True)

            try:
                response = self.llm.call_llm(VALIDATOR_SYSTEM_PROMPT, user_prompt)
                raw_content = response["choices"][0]["message"]["content"]
                parsed = json.loads(raw_content)
                evaluations = parsed.get("evaluations", []) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])
            except Exception as e:
                print(f"[Orchestrator] LLM evaluation notice: {e}. Falling back to heuristic scoring.", flush=True)

        # Step 3: Map evaluation scores and finalize results
        eval_map = {}
        for ev in evaluations:
            if "candidate_index" in ev:
                eval_map[ev["candidate_index"]] = ev
            if "name" in ev:
                eval_map[str(ev["name"]).strip().lower()] = ev

        valid_places = []
        for orig_idx, place in top_candidates:
            name_key = str(place.get("name", "")).strip().lower()
            ev = eval_map.get(orig_idx) or eval_map.get(name_key)

            if ev and "confidence_score" in ev:
                try:
                    score = float(ev["confidence_score"])
                except (ValueError, TypeError):
                    score = place.get("heuristic_score", 0.70)
                reason = ev.get("reason", "LLM validated.")
            else:
                score = place.get("heuristic_score", 0.70)
                reason = "Heuristic match."

            score = min(1.0, max(0.0, round(score, 2)))
            place["confidence_score"] = score
            place["confidence"] = f"{int(score * 100)}%"
            place["evaluation_reason"] = reason

            title = self._clean_ascii(place.get("name", "Unknown"))
            country_label = self._clean_ascii(place.get('country', 'N/A'))
            clean_reason = self._clean_ascii(reason)
            if score >= min_confidence:
                print(f"[Orchestrator] Candidate #{orig_idx} '{title}' ({country_label}) -> Confidence: {place['confidence']} (PASSED >= {int(min_confidence * 100)}%) | Reason: {clean_reason}", flush=True)
                valid_places.append(place)
            else:
                print(f"[Orchestrator] Candidate #{orig_idx} '{title}' ({country_label}) -> Confidence: {place['confidence']} (BELOW {int(min_confidence * 100)}%) | Reason: {clean_reason}", flush=True)

        # Adaptive Relaxation: If fewer than 5 candidates passed strictly at min_confidence,
        # gracefully accept high-quality combination matches (score >= 0.50) while strictly preserving location integrity
        if len(valid_places) < 5:
            print(f"[Orchestrator] Adaptive Mode: Only {len(valid_places)} strictly passed. Adding top combination matches (score >= 50%)...", flush=True)
            valid_names = {p.get("name", "").strip().lower() for p in valid_places}
            for orig_idx, place in top_candidates:
                p_name = place.get("name", "").strip().lower()
                p_score = place.get("confidence_score", 0.0)
                # Ensure location penalty candidates (<= 0.15) are NEVER added
                if p_name not in valid_names and p_score >= 0.50:
                    valid_names.add(p_name)
                    valid_places.append(place)
                    t_name = self._clean_ascii(place.get("name", "Unknown"))
                    t_country = self._clean_ascii(place.get('country', 'N/A'))
                    print(f"[Orchestrator] Accepted combination candidate '{t_name}' ({t_country}) -> Confidence: {place['confidence']}", flush=True)
                    if len(valid_places) >= 8:
                        break

        # Sort valid places by confidence score descending
        valid_places.sort(key=lambda x: x.get("confidence_score", 0.0), reverse=True)

        if valid_places:
            print(f"[Orchestrator] Validity check: {len(valid_places)} matched.\n", flush=True)
            return True, valid_places
        else:
            print(f"[Orchestrator] Validity check: No candidate passed matching threshold.\n", flush=True)
            return False, []

    def process_request_structured(self, user_input: str, target_count: int = 10) -> dict:
        """
        Full Hybrid Orchestration Workflow returning structured dict (used by FastAPI and CLI).
        """
        print(f"\n--- New Request: '{user_input}' ---")

        # Step 1: Extract preferences
        preferences = self.extractor.extract(user_input)

        # Step 2: Call Database Agent (Chroma DB + LlamaIndex)
        db_results = self.db_agent.search_database(preferences, top_k=target_count)

        # Step 3: Check validity of database results in batch
        is_valid_db, valid_db_places = self.check_validity(db_results, preferences, user_input, min_confidence=0.85)

        for p in valid_db_places:
            p["source"] = "Local Database"

        combined_places = list(valid_db_places)
        seen_names = {p.get("name", "").strip().lower() for p in combined_places}

        # Step 4: Hybrid Mode - If less than target_count places, search online for more
        if len(combined_places) < target_count:
            needed = target_count - len(combined_places)
            print(f"[Orchestrator] Hybrid Mode: Found {len(combined_places)} local database match(es). Searching online for {needed} additional destinations worldwide...", flush=True)

            search_candidates = self.search_agent.search_online(preferences, max_results=max(12, needed * 2))

            # Deduplicate against existing local places
            fresh_search_candidates = [
                p for p in search_candidates
                if p.get("name", "").strip().lower() not in seen_names
            ]

            if fresh_search_candidates:
                # Step 5: Batch evaluate search candidates
                is_valid_search, valid_search_places = self.check_validity(fresh_search_candidates, preferences, user_input, min_confidence=0.70)

                # Step 6: Automatically save newly discovered search destinations into ChromaDB for future queries
                if valid_search_places:
                    self.db_agent.insert_places(valid_search_places)

                for p in valid_search_places:
                    p["source"] = "Web / Wikipedia Search (Now Saved to DB)"
                    if p.get("name", "").strip().lower() not in seen_names:
                        seen_names.add(p.get("name", "").strip().lower())
                        combined_places.append(p)
                        if len(combined_places) >= target_count:
                            break

        # Step 7: Sort by confidence score descending
        combined_places.sort(key=lambda x: x.get("confidence_score", 0.0), reverse=True)
        final_places = combined_places[:target_count]

        source_label = "Hybrid (Local Database + Global Search with Auto-DB Ingestion)" if len(final_places) > len(valid_db_places) else "Local Database"
        return {
            "success": len(final_places) > 0,
            "user_query": user_input,
            "preferences": preferences,
            "places": final_places,
            "total_places": len(final_places),
            "source": source_label,
            "formatted_text": self.format_response(final_places, source=source_label) if final_places else "Sorry, I can't work on this request as no destination in our database or online search matched your exact preferences. Please try refining your request!"
        }

    def process_request(self, user_input: str, target_count: int = 10) -> str:
        res = self.process_request_structured(user_input, target_count=target_count)
        return res["formatted_text"]

    def _clean_ascii(self, text: str) -> str:
        if not text:
            return ""
        return str(text).encode("ascii", errors="replace").decode("ascii").replace("?", "")

    def format_response(self, places: list, source: str) -> str:
        """
        Formats recommendations nicely for user display.
        """
        lines = [f"=== Tourism Recommendations (Source: {source} | Total: {len(places)} Places) ==="]
        for idx, place in enumerate(places, 1):
            name = self._clean_ascii(place.get('name', 'Unknown'))
            place_src = place.get("source", "Hybrid")
            lines.append(f"\n[Place #{idx}] {name} ({place_src})")
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
            if len(desc) > 280:
                desc = desc[:280] + "..."
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
        demo_prompt = "I want to go on a relaxing beach where i can see the sunset and drink some wine go to a pub and have a relaxing time in Australia"
        print(f"Running Demo Query: '{demo_prompt}'")
        res = orchestrator.process_request(demo_prompt)
        print("\n" + res)
