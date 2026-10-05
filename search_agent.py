import json
import re
import warnings
import wikipedia
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts import SEARCH_ENRICHMENT_SYSTEM_PROMPT, build_enrichment_user_prompt


class SearchAgent:
    """
    Search Agent (Wikipedia & Real Destination Search with GPS Enrichment):
    - Searches Wikipedia for real geographical destinations (beaches, islands, parks, towns).
    - Auto-extracts exact GPS coordinates (latitude, longitude), country, state, locality, vibe, and activities.
    - Resolves disambiguation pages and extracts high-resolution images.
    - Guarantees every returned candidate conforms to the complete database destination schema.
    """

    def __init__(self, llm=None):
        self.llm = llm
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from duckduckgo_search import DDGS
                self.ddgs = DDGS()
            self.ddg_enabled = True
        except Exception as e:
            print(f"[SearchAgent] DDGS initialization notice: {e}", flush=True)
            self.ddg_enabled = False

    def auto_extract_metadata(self, title: str, content: str, page_obj=None, preferences: dict = None) -> dict:
        """
        Auto-extracts missing metadata (latitude, longitude, country, state, locality) from Wikipedia page/text.
        """
        text = f"{title} {content}"

        # 1. GPS Coordinates from Wikipedia page object
        lat, lon = 0.0, 0.0
        if page_obj:
            try:
                coords = getattr(page_obj, "coordinates", None)
                if coords and len(coords) >= 2:
                    lat = float(coords[0])
                    lon = float(coords[1])
            except Exception:
                lat, lon = 0.0, 0.0

        # 2. Country extraction
        user_country = ", ".join([c for c in preferences.get("country_names", []) if c.strip()]) if preferences else ""
        if user_country:
            country = user_country
        else:
            common_countries = [
                "Australia", "Japan", "United States", "USA", "France", "Italy", "United Kingdom", "UK",
                "China", "Denmark", "Bangladesh", "Malaysia", "Canada", "Germany", "Spain", "Greece",
                "Indonesia", "Thailand", "New Zealand", "Portugal", "Croatia", "Fiji", "Maldives", "Mexico", "Brazil"
            ]
            detected = [c for c in common_countries if re.search(rf"\b{re.escape(c)}\b", text, re.IGNORECASE)]
            country = detected[0] if detected else "International"

        # 3. State extraction
        common_states = [
            "New South Wales", "Victoria", "Queensland", "California", "Hawaii", "Tokyo Metropolis",
            "Bavaria", "Île-de-France", "Tasmania", "Western Australia", "South Australia", "Bali",
            "Phuket", "Santorini", "Cyclades", "Amalfi Coast", "Canary Islands", "Baleares", "Rio de Janeiro"
        ]
        detected_states = [s for s in common_states if re.search(rf"\b{re.escape(s)}\b", text, re.IGNORECASE)]
        state = detected_states[0] if detected_states else ""

        # 4. Locality/City extraction
        user_cities = [ct for ct in preferences.get("city_names", []) if ct.strip()] if preferences else []
        if user_cities:
            locality = user_cities
        else:
            common_cities = [
                "Sydney", "Bondi", "Manly", "Byron Bay", "Melbourne", "Brisbane", "Perth", "Cairns",
                "Tokyo", "Paris", "London", "Rome", "Lithgow", "Katoomba", "Copenhagen", "Beijing",
                "New York", "Honolulu", "Waikiki", "Nice", "Cannes", "Ibiza", "Mykonos", "Denpasar", "Patong"
            ]
            detected_cities = [ct for ct in common_cities if re.search(rf"\b{re.escape(ct)}\b", text, re.IGNORECASE)]
            locality = list(dict.fromkeys(detected_cities)) if detected_cities else []

        return {
            "latitude": lat,
            "longitude": lon,
            "country": country,
            "state": state,
            "locality": locality
        }

    def evaluate_confidence(self, title: str, summary: str, preferences: dict) -> float:
        """
        Smarter Combinatorial Scoring with Strict Location Enforcement:
        1. Location Penalty: If user specifies country/city and candidate is elsewhere, apply heavy penalty (cap <= 0.15).
        2. Combinatorial Keyword Match: Rewards candidates matching combinations of (Interests, Activities, Vibes, and Keywords).
        3. Graceful Partial Matches: Evaluates best combinations (e.g. mountain+camping, camping+warm, sunset+camping).
        """
        text = f"{title} {summary}".lower()

        countries = [c.lower() for c in preferences.get("country_names", []) if c.strip()]
        cities = [c.lower() for c in preferences.get("city_names", []) if c.strip()]
        interests = [i.lower() for i in preferences.get("place_of_interests", []) if i.strip()]
        vibes = [v.lower() for v in preferences.get("vibe", []) if v.strip()]
        activities = [a.lower() for a in preferences.get("place_of_activites", []) if a.strip()]
        specifics = [s.lower() for s in preferences.get("specific_named_places", []) if s.strip()]
        raw_input = preferences.get("raw_user_input", "").lower()

        # ----------------------------------------------------
        # 1. Location Constraint & Hard Penalty
        # ----------------------------------------------------
        location_score = 0.0
        if countries or cities:
            matched_loc = False
            if countries and any(c in text for c in countries):
                matched_loc = True
            if cities and any(ct in text for ct in cities):
                matched_loc = True
            if not matched_loc:
                # Severe location mismatch penalty: hard-cap score
                return 0.15
            location_score = 0.30
        else:
            # Neutral credit for worldwide open requests
            location_score = 0.30

        # ----------------------------------------------------
        # 2. Specific Named Place Match (e.g. Bondi, Mount Fuji)
        # ----------------------------------------------------
        specific_score = 0.0
        if specifics:
            if any(sp in text for sp in specifics):
                specific_score = 0.25
        else:
            specific_score = 0.10

        # ----------------------------------------------------
        # 3. Combinatorial Feature & Keyword Evaluation
        # ----------------------------------------------------
        # Collect distinct feature pools
        matched_interests = [i for i in interests if i in text]
        matched_vibes = [v for v in vibes if v in text]
        matched_activities = [a for a in activities if a in text]

        # Extract meaningful words from raw prompt (e.g., warm, sunset, camping, mountain, weather)
        stop_words = {"want", "would", "like", "with", "from", "that", "this", "have", "some", "good", "place", "where", "into", "onto", "top"}
        raw_tokens = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", raw_input) if w not in stop_words]
        matched_raw_tokens = [w for w in raw_tokens if w in text]

        # Base interest match
        interest_score = 0.0
        if interests:
            if matched_interests:
                interest_score = 0.25
            else:
                interest_score = 0.05
        else:
            interest_score = 0.15

        # Activity & Vibe match
        act_vibe_score = 0.0
        total_act_vibe = activities + vibes
        if total_act_vibe:
            matched_act_vibe = matched_activities + matched_vibes
            act_vibe_ratio = len(matched_act_vibe) / max(1, len(total_act_vibe))
            act_vibe_score = min(0.20, act_vibe_ratio * 0.20 + (0.08 if matched_act_vibe else 0.0))
        else:
            act_vibe_score = 0.10

        # Combination Synergy Bonus: reward when 2 or more distinct dimensions co-occur
        combo_bonus = 0.0
        matched_categories = (1 if matched_interests else 0) + (1 if matched_activities else 0) + (1 if matched_vibes else 0) + (1 if len(matched_raw_tokens) >= 2 else 0)
        if matched_categories >= 3:
            combo_bonus = 0.15  # Strong multi-feature match (e.g. Mountain + Camping + Sunset)
        elif matched_categories == 2:
            combo_bonus = 0.10  # Good combination match (e.g. Mountain + Camping or Camping + Warm)
        elif matched_raw_tokens:
            combo_bonus = min(0.08, len(matched_raw_tokens) * 0.03)

        total_score = location_score + specific_score + interest_score + act_vibe_score + combo_bonus
        return min(1.0, max(0.10, round(total_score, 2)))

    def _fetch_wikipedia_place(self, candidate_title: str, preferences: dict, min_confidence: float = 0.50) -> dict | None:
        """
        Fetches a single Wikipedia page, handling disambiguation and auto-extracting full place metadata.
        """
        try:
            page_obj = None
            try:
                page_obj = wikipedia.page(candidate_title, auto_suggest=False)
            except wikipedia.exceptions.DisambiguationError as dis_err:
                for opt in dis_err.options[:3]:
                    try:
                        page_obj = wikipedia.page(opt, auto_suggest=False)
                        break
                    except Exception:
                        continue
            except Exception:
                try:
                    page_obj = wikipedia.page(candidate_title, auto_suggest=True)
                except Exception:
                    return None

            if not page_obj:
                return None

            title = self._clean_ascii(getattr(page_obj, "title", candidate_title))
            summary = self._clean_ascii(getattr(page_obj, "summary", ""))
            if len(summary) > 500:
                summary = summary[:500] + "..."

            confidence = self.evaluate_confidence(title, summary, preferences)
            if confidence < min_confidence:
                return None

            metadata = self.auto_extract_metadata(title, summary, page_obj=page_obj, preferences=preferences)

            image_url = ""
            if hasattr(page_obj, "images") and page_obj.images:
                for img in page_obj.images:
                    if img.endswith((".jpg", ".png", ".jpeg", ".webp")) and not any(ic in img.lower() for ic in ["icon", "logo", "flag", "symbol", "stub", "svg"]):
                        image_url = img
                        break

            return {
                "name": title,
                "text_content": summary,
                "latitude": float(metadata.get("latitude", 0.0)),
                "longitude": float(metadata.get("longitude", 0.0)),
                "place_of_interests": preferences.get("place_of_interests", []) or ["tourism", "nature"],
                "vibe": preferences.get("vibe", ["relaxing"]),
                "place_of_activites": preferences.get("place_of_activites", ["sightseeing"]),
                "country": metadata.get("country", "International"),
                "state": metadata.get("state", ""),
                "locality": metadata.get("locality", []),
                "image_url": image_url,
                "confidence": f"{int(confidence * 100)}%",
                "confidence_score": confidence,
                "source_url": page_obj.url if hasattr(page_obj, "url") else f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            }
        except Exception as e:
            print(f"[SearchAgent] Notice fetching '{candidate_title}': {e}", flush=True)
            return None

    def _clean_ascii(self, text: str) -> str:
        if not text:
            return ""
        return str(text).encode("ascii", errors="replace").decode("ascii").replace("?", "")

    def search_wikipedia(self, preferences: dict, max_results: int = 10, min_confidence: float = 0.50) -> list:
        """
        Searches Wikipedia based on preferences to discover genuine destinations.
        Uses parallel thread pool fetching for high speed.
        """
        country = ", ".join([c for c in preferences.get("country_names", []) if c.strip()])
        city = ", ".join([c for c in preferences.get("city_names", []) if c.strip()])
        interests = [i for i in preferences.get("place_of_interests", []) if i.strip()]
        interest_term = interests[0] if interests else "tourist attractions"
        specifics = ", ".join([s for s in preferences.get("specific_named_places", []) if s.strip()])
        raw_input = preferences.get("raw_user_input", "")

        clean_input = " ".join([w for w in raw_input.split() if w.lower() not in ["i", "want", "a", "to", "go", "on", "in", "and", "the", "where", "can", "see", "some", "have", "with"]])

        # Curate destination search queries dynamically based on actual user interests
        search_queries = []
        if specifics:
            search_queries.append(f"{specifics} {city} {country}")
        if city or country:
            search_queries.append(f"{interest_term} in {city} {country}".strip())
            search_queries.append(f"{city} {country} attractions".strip())
            search_queries.append(f"famous places in {city} {country}".strip())
        else:
            # Broad worldwide queries based on user's target interest
            search_queries.append(f"famous {interest_term} in Australia")
            search_queries.append(f"popular {interest_term} destinations")
            search_queries.append(f"{clean_input} travel destinations")

        all_candidate_titles = []
        seen_titles = set()

        bad_title_tokens = [
            "list of", "history of", "economy of", "geography of", "demographics of", "index of", "outline of", "portal:",
            "extermination", "concentration camp", "visitor center", "tourist sign", "campsite", "camping world",
            "air lines", "noctourism", "tourism in ", "scout ranch"
        ]

        # Step 1: Collect titles from Wikipedia search
        for query_str in search_queries:
            query_str = query_str.strip()
            if not query_str:
                continue

            print(f"[SearchAgent] Searching Wikipedia for: '{self._clean_ascii(query_str)}'", flush=True)
            try:
                candidate_titles = wikipedia.search(query_str, results=6)
                for title in candidate_titles:
                    if title.lower() in seen_titles:
                        continue
                    seen_titles.add(title.lower())

                    # Skip non-place articles
                    if any(bad in title.lower() for bad in bad_title_tokens):
                        continue
                    all_candidate_titles.append(title)
            except Exception as e:
                print(f"[SearchAgent] Wikipedia query error: {self._clean_ascii(str(e))}", flush=True)

        # Step 2: Fetch pages in PARALLEL using ThreadPoolExecutor
        results = []
        if all_candidate_titles:
            print(f"[SearchAgent] Fetching {len(all_candidate_titles)} Wikipedia pages concurrently in parallel...", flush=True)
            with ThreadPoolExecutor(max_workers=6) as executor:
                future_to_title = {
                    executor.submit(self._fetch_wikipedia_place, title, preferences, min_confidence): title
                    for title in all_candidate_titles
                }
                for future in as_completed(future_to_title):
                    try:
                        place = future.result()
                        if place:
                            results.append(place)
                            clean_name = self._clean_ascii(place['name'])
                            clean_country = self._clean_ascii(place['country'])
                            print(f"[SearchAgent] Discovered Destination '{clean_name}' ({clean_country}) -> Score: {place['confidence']}", flush=True)
                            if len(results) >= max_results:
                                break
                    except Exception as e:
                        pass

        # Step 3: Combinatorial Fallback if results < 5
        # If strict search returned fewer than 5 places, search best keyword combinations (e.g. mountain+camping, camping+warm)
        if len(results) < 5:
            print(f"[SearchAgent] Only {len(results)} candidate(s) found. Triggering Combinatorial Keyword Fallback...", flush=True)
            activities = [a for a in preferences.get("place_of_activites", []) if a.strip()]
            vibes = [v for v in preferences.get("vibe", []) if v.strip()]
            loc_suffix = f" in {city} {country}".strip() if (city or country) else ""

            fallback_combos = []
            for act in activities:
                if interests:
                    fallback_combos.append(f"{interests[0]} {act}{loc_suffix}")
                fallback_combos.append(f"{act} camping destinations{loc_suffix}")
                fallback_combos.append(f"{act} tourist destinations{loc_suffix}")
            for v in vibes:
                fallback_combos.append(f"{v} {interest_term}{loc_suffix}")
            
            # Additional prompt keywords (e.g. camping, sunset, mountain, warm)
            fallback_combos.append(f"camping national parks{loc_suffix}")
            fallback_combos.append(f"scenic mountain camping{loc_suffix}")

            fallback_titles = []
            for combo_q in fallback_combos:
                if len(results) + len(fallback_titles) >= max_results * 2:
                    break
                combo_q = combo_q.strip()
                if not combo_q:
                    continue
                try:
                    c_titles = wikipedia.search(combo_q, results=4)
                    for t in c_titles:
                        if t.lower() not in seen_titles and not any(bad in t.lower() for bad in bad_title_tokens):
                            seen_titles.add(t.lower())
                            fallback_titles.append(t)
                except Exception:
                    pass

            if fallback_titles:
                print(f"[SearchAgent] Fetching {len(fallback_titles)} combinatorial fallback pages concurrently...", flush=True)
                with ThreadPoolExecutor(max_workers=6) as executor:
                    future_to_title = {
                        # Slightly lower threshold for combinatorial fallback so partial matches are evaluated
                        executor.submit(self._fetch_wikipedia_place, title, preferences, 0.45): title
                        for title in fallback_titles
                    }
                    for future in as_completed(future_to_title):
                        try:
                            place = future.result()
                            if place:
                                results.append(place)
                                clean_name = self._clean_ascii(place['name'])
                                print(f"[SearchAgent] Fallback Combination Match: '{clean_name}' -> Score: {place['confidence']}", flush=True)
                                if len(results) >= max_results:
                                    break
                        except Exception:
                            pass

        # Sort results by confidence score descending
        results.sort(key=lambda x: x.get("confidence_score", 0.0), reverse=True)
        return results

    def search_online(self, preferences: dict, max_results: int = 10) -> list:
        """
        Main entry point for online search:
        Retrieves real destination places from Wikipedia.
        """
        wiki_results = self.search_wikipedia(preferences, max_results=max_results, min_confidence=0.65)
        return wiki_results


if __name__ == "__main__":
    agent = SearchAgent()
    sample_prefs = {
        "country_names": ["Australia"],
        "city_names": [""],
        "place_of_interests": ["beach", "sunset", "wine"],
        "place_of_activites": ["drinking wine", "going to a pub"],
        "vibe": ["relaxing", "chill"],
        "raw_user_input": "I want to go on a relaxing beach where i can see the sunset and drink some wine go to a pub and have a relaxing time"
    }
    res = agent.search_online(sample_prefs, max_results=5)
    print(json.dumps(res, indent=2))
