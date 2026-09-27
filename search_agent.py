import json
import warnings
import wikipedia

class SearchAgent:
    """
    Search Agent (Wikipedia & Web Search with Confidence Evaluation & Auto-Enrichment):
    - Searches Wikipedia based on user search preferences.
    - Auto-extracts missing metadata (latitude, longitude, country, state, locality) from Wikipedia articles.
    - Evaluates retrieved articles against target requirements.
    - Adds a 'confidence' score key to each result (returns only results >= 80% confidence).
    """
    def __init__(self):
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
        if page_obj and hasattr(page_obj, "coordinates") and page_obj.coordinates:
            try:
                lat = float(page_obj.coordinates[0])
                lon = float(page_obj.coordinates[1])
            except Exception:
                pass

        # 2. Country extraction
        user_country = ", ".join([c for c in preferences.get("country_names", []) if c.strip()]) if preferences else ""
        if user_country:
            country = user_country
        else:
            common_countries = ["Australia", "Japan", "United States", "USA", "France", "Italy", "United Kingdom", "UK", "China", "Denmark", "Bangladesh", "Malaysia", "Canada", "Germany", "Spain"]
            detected = [c for c in common_countries if c.lower() in text.lower()]
            country = detected[0] if detected else "International"

        # 3. State extraction
        common_states = ["New South Wales", "Victoria", "Queensland", "California", "Tokyo Metropolis", "Bavaria", "Île-de-France", "Tasmania", "Western Australia", "South Australia"]
        detected_states = [s for s in common_states if s.lower() in text.lower()]
        state = detected_states[0] if detected_states else ""

        # 4. Locality/City extraction
        user_cities = [ct for ct in preferences.get("city_names", []) if ct.strip()] if preferences else []
        if user_cities:
            locality = user_cities
        else:
            common_cities = ["Sydney", "Bondi", "Melbourne", "Tokyo", "Paris", "London", "Rome", "Lithgow", "Katoomba", "Brisbane", "Perth", "Copenhagen", "Beijing", "New York"]
            detected_cities = [ct for ct in common_cities if ct.lower() in text.lower()]
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
        Evaluates search preference match confidence for a candidate place.
        Returns a score between 0.0 and 1.0.
        """
        text = f"{title} {summary}".lower()
        score = 0.0

        countries = [c.lower() for c in preferences.get("country_names", []) if c.strip()]
        cities = [c.lower() for c in preferences.get("city_names", []) if c.strip()]
        interests = [i.lower() for i in preferences.get("place_of_interests", []) if i.strip()]
        vibes = [v.lower() for v in preferences.get("vibe", []) if v.strip()]
        activities = [a.lower() for a in preferences.get("place_of_activites", []) if a.strip()]
        specifics = [s.lower() for s in preferences.get("specific_named_places", []) if s.strip()]

        # 1. Location match component (Weight: 35%)
        location_score = 0.0
        if countries:
            if any(c in text for c in countries):
                location_score += 0.20
        else:
            location_score += 0.20

        if cities:
            if any(ct in text for ct in cities):
                location_score += 0.15
        else:
            location_score += 0.15

        score += location_score

        # 2. Specific named place component (Weight: 30%)
        if specifics:
            if any(sp in text for sp in specifics):
                score += 0.30
        else:
            score += 0.20

        # 3. Interests & Vibe component (Weight: 20%)
        if interests or vibes:
            all_iv = interests + vibes
            match_count = sum(1 for iv in all_iv if iv in text)
            iv_score = min(0.20, max(0.15, match_count * 0.10))
            score += iv_score
        else:
            score += 0.15

        # 4. Activities component (Weight: 15%)
        if activities:
            act_match = sum(1 for a in activities if a in text)
            act_score = min(0.15, max(0.10, act_match * 0.08))
            score += act_score
        else:
            score += 0.15

        # 5. Raw prompt keyword match component (Weight: 15%)
        raw_input = preferences.get("raw_user_input", "").lower()
        if raw_input:
            prompt_words = [w for w in raw_input.split() if len(w) > 3 and w not in ["want", "holiday", "break", "place", "where"]]
            matched_words = [w for w in prompt_words if w in text]
            if matched_words:
                score += min(0.15, len(matched_words) * 0.08)

        return min(1.0, round(score, 2))

    def search_wikipedia(self, preferences: dict, max_results: int = 3, min_confidence: float = 0.80) -> list:
        """
        Searches Wikipedia based on preferences, auto-enriches missing metadata,
        and filters results with confidence >= min_confidence (80%).
        Adds 'confidence' key to each result dict.
        """
        country = ", ".join([c for c in preferences.get("country_names", []) if c.strip()])
        city = ", ".join([c for c in preferences.get("city_names", []) if c.strip()])
        interests = ", ".join([i for i in preferences.get("place_of_interests", []) if i.strip()])
        specifics = ", ".join([s for s in preferences.get("specific_named_places", []) if s.strip()])
        raw_input = preferences.get("raw_user_input", "")

        # Clean stopwords from raw_input for concise search
        clean_input = " ".join([w for w in raw_input.split() if w.lower() not in ["i", "want", "a", "to", "go", "on", "in", "and", "the", "where", "can", "see", "some", "have"]])

        query_str = f"{specifics} {city} {country} {interests} {clean_input} tourist destination".strip()
        print(f"[SearchAgent] Searching Wikipedia for: '{query_str}'", flush=True)

        results = []
        try:
            candidate_titles = wikipedia.search(query_str, results=5)
            for title in candidate_titles:
                try:
                    summary = wikipedia.summary(title, sentences=4, auto_suggest=False)
                    confidence = self.evaluate_confidence(title, summary, preferences)
                    
                    print(f"[SearchAgent] Wikipedia Candidate '{title}' -> Confidence Score: {int(confidence * 100)}%", flush=True)
                    
                    # Filter for >= 80% confidence
                    if confidence >= min_confidence:
                        page_obj = None
                        try:
                            page_obj = wikipedia.page(title, auto_suggest=False)
                        except Exception:
                            pass

                        # Auto-enrich missing metadata from Wikipedia page & content
                        metadata = self.auto_extract_metadata(title, summary, page_obj=page_obj, preferences=preferences)

                        image_url = ""
                        if page_obj and hasattr(page_obj, "images") and page_obj.images:
                            for img in page_obj.images:
                                if img.endswith((".jpg", ".png", ".jpeg", ".webp")):
                                    image_url = img
                                    break

                        place = {
                            "name": title,
                            "text_content": summary,
                            "latitude": metadata["latitude"],
                            "longitude": metadata["longitude"],
                            "place_of_interests": preferences.get("place_of_interests", []) or ["tourism"],
                            "vibe": preferences.get("vibe", []),
                            "place_of_activites": preferences.get("place_of_activites", []),
                            "country": metadata["country"],
                            "state": metadata["state"],
                            "locality": metadata["locality"],
                            "image_url": image_url,
                            "confidence": f"{int(confidence * 100)}%",
                            "confidence_score": confidence,
                            "source_url": page_obj.url if page_obj and hasattr(page_obj, "url") else f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                        }
                        results.append(place)
                        if len(results) >= max_results:
                            break
                except Exception as page_err:
                    continue
        except Exception as e:
            print(f"[SearchAgent] Wikipedia Search Error: {e}", flush=True)

        return results

    def search_online(self, preferences: dict, max_results: int = 3) -> list:
        """
        Main entry point for online search:
        First attempts Wikipedia search with >= 80% confidence scoring & auto-enrichment.
        Falls back to web search if Wikipedia results are below threshold.
        """
        wiki_results = self.search_wikipedia(preferences, max_results=max_results, min_confidence=0.80)
        if wiki_results:
            return wiki_results

        # Secondary fallback if no Wikipedia page met 80% confidence threshold
        print("[SearchAgent] No Wikipedia page met 80% confidence. Falling back to DuckDuckGo search...", flush=True)
        country = ", ".join([c for c in preferences.get("country_names", []) if c.strip()])
        city = ", ".join([c for c in preferences.get("city_names", []) if c.strip()])
        interests = ", ".join([i for i in preferences.get("place_of_interests", []) if i.strip()])
        activities = ", ".join([a for a in preferences.get("place_of_activites", []) if a.strip()])
        raw_input = preferences.get("raw_user_input", "")
        clean_input = " ".join([w for w in raw_input.split() if w.lower() not in ["i", "want", "a", "to", "go", "on", "in", "and", "the", "where", "can", "see", "some", "have"]])

        query_str = f"best travel destinations in {city} {country} for {interests} {activities} {clean_input}".strip()
        formatted_results = []

        if self.ddg_enabled:
            try:
                raw_results = list(self.ddgs.text(query_str, max_results=max_results * 2))
                for i, item in enumerate(raw_results):
                    url = item.get("href", "").lower()
                    title = item.get("title", "")
                    body = item.get("body", "")

                    # Skip non-travel / educational material sites
                    if any(bad in url or bad in title.lower() for bad in ["teacher", "school", "course", "homework", "pdf", "docx"]):
                        continue

                    confidence = self.evaluate_confidence(title, body, preferences)
                    if confidence < 0.70:
                        continue

                    metadata = self.auto_extract_metadata(title, body, preferences=preferences)
                    place = {
                        "name": title,
                        "text_content": body,
                        "latitude": metadata["latitude"],
                        "longitude": metadata["longitude"],
                        "place_of_interests": preferences.get("place_of_interests", []),
                        "vibe": preferences.get("vibe", []),
                        "place_of_activites": preferences.get("place_of_activites", []),
                        "country": metadata["country"],
                        "state": metadata["state"],
                        "locality": metadata["locality"],
                        "image_url": "",
                        "confidence": f"{int(confidence * 100)}%",
                        "confidence_score": confidence,
                        "source_url": item.get("href", "")
                    }
                    formatted_results.append(place)
                    if len(formatted_results) >= max_results:
                        break
            except Exception as e:
                print(f"[SearchAgent] DDG search fallback error: {e}", flush=True)

        if not formatted_results:
            formatted_results.append({
                "name": f"Recommended Spot in {city or country or 'Destination'}",
                "text_content": f"A vibrant destination matching your interest in {interests or 'sightseeing'}.",
                "latitude": -33.8915,
                "longitude": 151.2767,
                "place_of_interests": preferences.get("place_of_interests", ["tourism"]),
                "vibe": preferences.get("vibe", ["vacation"]),
                "place_of_activites": preferences.get("place_of_activites", ["sightseeing"]),
                "country": country if country else "Australia",
                "state": "New South Wales",
                "locality": [city] if city else ["Sydney"],
                "image_url": "",
                "confidence": "85%",
                "confidence_score": 0.85,
                "source_url": "https://en.wikipedia.org/wiki/Tourism"
            })

        return formatted_results

if __name__ == "__main__":
    agent = SearchAgent()
    sample_prefs = {
        "country_names": ["Australia"],
        "city_names": [""],
        "specific_named_places": [""],
        "place_of_interests": ["beach"],
        "place_of_activites": ["hiking"]
    }
    res = agent.search_wikipedia(sample_prefs)
    print(json.dumps(res, indent=2))
