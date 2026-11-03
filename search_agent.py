import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts import (
    FULL_SOURCE_SUMMARIZER_PROMPT,
    build_full_source_summarizer_user_prompt
)


class SearchAgent:
    """
    Search Agent (Wikivoyage & Wikipedia Travel Destination Entity Engine):
    - Problem 1: Hybrid semantic + keyword retrieval targeting joint features (e.g. mountain near beach).
    - Problem 2: Entity searching via Wikivoyage API (curated travel) and Wikipedia API (with entity type filtering).
    - Problem 4: Accurate LLM geographical extraction (country, state, locality) avoiding naive regex bugs.
    - Destination Descriptions: LLM reads full source text and writes an informative ~300-500 word travel summary.
    """

    def __init__(self, llm=None):
        self.llm = llm
        self.headers = {"User-Agent": "VoyageAI-TourismDiscovery/2.0 (travel-agent-bot)"}

    def _clean_ascii(self, text: str) -> str:
        if not text:
            return ""
        return str(text).encode("ascii", errors="replace").decode("ascii").replace("?", "")

    # =========================================================================
    # 1. WIKIVOYAGE API INTEGRATION (Best for Curated Travel Destinations)
    # =========================================================================

    def _search_wikivoyage(self, query_str: str, limit: int = 5) -> list:
        """
        Queries Wikivoyage search API for authentic travel destinations.
        """
        try:
            url = f"https://en.wikivoyage.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query_str)}&format=json"
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                search_results = data.get("query", {}).get("search", [])
                return [item["title"] for item in search_results[:limit] if not item["title"].startswith("Wikivoyage:")]
        except Exception as e:
            print(f"[SearchAgent] Wikivoyage query error: {self._clean_ascii(str(e))}", flush=True)
            return []

    def _fetch_wikivoyage_destination(self, title: str) -> dict | None:
        """
        Fetches full destination travel guide, coordinates, and thumbnail from Wikivoyage.
        """
        try:
            url = (
                f"https://en.wikivoyage.org/w/api.php?action=query&prop=extracts|coordinates|pageimages"
                f"&titles={urllib.parse.quote(title)}&explaintext=1&piprop=original|thumbnail&pithumbsize=800&format=json"
            )
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                for pid, p in pages.items():
                    if pid == "-1" or "missing" in p:
                        continue
                    extract = p.get("extract", "")
                    if len(extract) < 80:
                        continue
                    
                    coords = p.get("coordinates", [])
                    lat, lon = 0.0, 0.0
                    if coords and len(coords) > 0:
                        lat = float(coords[0].get("lat", 0.0))
                        lon = float(coords[0].get("lon", 0.0))

                    image_url = p.get("thumbnail", {}).get("source", "")
                    if not image_url and "original" in p:
                        image_url = p.get("original", {}).get("source", "")

                    source_url = f"https://en.wikivoyage.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                    return {
                        "name": title,
                        "raw_extract": extract,
                        "latitude": lat,
                        "longitude": lon,
                        "image_url": image_url,
                        "source_url": source_url,
                        "source_type": "Wikivoyage Travel Guide"
                    }
        except Exception as e:
            print(f"[SearchAgent] Notice fetching Wikivoyage '{title}': {self._clean_ascii(str(e))}", flush=True)
        return None

    # =========================================================================
    # 2. WIKIPEDIA API WITH STRICT ENTITY FILTERING (Excludes Businesses/Lists)
    # =========================================================================

    def _search_wikipedia(self, query_str: str, limit: int = 5) -> list:
        """
        Queries Wikipedia API for candidates.
        """
        try:
            url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query_str)}&format=json"
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                search_results = data.get("query", {}).get("search", [])
                return [item["title"] for item in search_results[:limit]]
        except Exception as e:
            print(f"[SearchAgent] Wikipedia query error: {self._clean_ascii(str(e))}", flush=True)
            return []

    def _fetch_wikipedia_entity_destination(self, title: str) -> dict | None:
        """
        Fetches Wikipedia page with description and pageprops to strictly filter out non-destinations
        (e.g. UK travel agencies, retailers, airlines, disambiguation pages, listicles).
        """
        try:
            url = (
                f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts|coordinates|pageimages|description|pageprops"
                f"&titles={urllib.parse.quote(title)}&explaintext=1&piprop=original|thumbnail&pithumbsize=800&format=json"
            )
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                for pid, p in pages.items():
                    if pid == "-1" or "missing" in p:
                        continue

                    # 1. Strict entity filter: check pageprops
                    pageprops = p.get("pageprops", {})
                    if "disambiguation" in pageprops:
                        return None

                    # 2. Strict entity filter: check description for corporate/non-place markers
                    desc = str(p.get("description", "")).lower()
                    title_lower = title.lower()
                    bad_entity_tokens = [
                        "business", "company", "retailer", "agency", "travel retailer", "airline",
                        "holding", "bank", "musician", "band", "album", "film", "song", "software",
                        "politician", "actor", "organisation", "organization"
                    ]
                    if any(bad in desc for bad in bad_entity_tokens):
                        print(f"[SearchAgent] Discarded non-destination entity '{title}' ({desc})", flush=True)
                        return None

                    # Discard meta listicles
                    if any(title_lower.startswith(prefix) for prefix in ["list of", "index of", "outline of", "portal:", "history of"]):
                        return None

                    extract = p.get("extract", "")
                    if len(extract) < 80:
                        return None

                    coords = p.get("coordinates", [])
                    lat, lon = 0.0, 0.0
                    if coords and len(coords) > 0:
                        lat = float(coords[0].get("lat", 0.0))
                        lon = float(coords[0].get("lon", 0.0))

                    image_url = p.get("thumbnail", {}).get("source", "")
                    if not image_url and "original" in p:
                        image_url = p.get("original", {}).get("source", "")

                    source_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                    return {
                        "name": title,
                        "raw_extract": extract,
                        "short_desc": p.get("description", ""),
                        "latitude": lat,
                        "longitude": lon,
                        "image_url": image_url,
                        "source_url": source_url,
                        "source_type": "Wikipedia Verified Entity"
                    }
        except Exception as e:
            print(f"[SearchAgent] Notice fetching Wikipedia '{title}': {self._clean_ascii(str(e))}", flush=True)
        return None

    # =========================================================================
    # 3. LLM FULL SOURCE READING, SUMMARIZATION (~300-500 Words) & METADATA
    # =========================================================================

    def summarize_and_extract_metadata(self, title: str, raw_extract: str, preferences: dict, short_desc: str = "", use_llm: bool = False) -> dict | None:
        """
        Extracts clean metadata and summary.
        If use_llm=False (default during initial search discovery): uses instant extractive lead to avoid blocking multi-minute LLM generation.
        If use_llm=True: generates the full in-depth ~300-500 word AI guide using LLM.
        """
        user_context = f"{', '.join(preferences.get('place_of_interests', []))} {', '.join(preferences.get('place_of_activites', []))} {preferences.get('raw_user_input', '')}".strip()

        # Step 1: Use LLM if explicitly requested
        if use_llm and self.llm is not None and getattr(self.llm, "llm", None) is not None:
            try:
                user_prompt = build_full_source_summarizer_user_prompt(title, raw_extract, user_context)
                response = self.llm.call_llm(FULL_SOURCE_SUMMARIZER_PROMPT, user_prompt, max_tokens=650)
                raw_json = response["choices"][0]["message"]["content"]
                parsed = json.loads(raw_json)

                if parsed.get("is_real_destination") is False:
                    print(f"[SearchAgent] LLM flagged '{title}' as non-real destination. Dropping.", flush=True)
                    return None

                summary = parsed.get("summary_500_words") or parsed.get("summary") or ""
                if len(summary.split()) < 40 and raw_extract:
                    summary = f"{summary}\n\n{raw_extract[:1200]}".strip()

                return {
                    "summary": summary,
                    "country": parsed.get("country") or "International",
                    "state": parsed.get("state") or "",
                    "locality": parsed.get("locality") or [],
                    "place_of_interests": parsed.get("place_of_interests") or preferences.get("place_of_interests", ["nature"]),
                    "vibe": parsed.get("vibe") or preferences.get("vibe", ["scenic"]),
                    "place_of_activites": parsed.get("place_of_activites") or preferences.get("place_of_activites", ["sightseeing"])
                }
            except Exception as e:
                print(f"[SearchAgent] LLM summarization notice for '{title}': {self._clean_ascii(str(e))}. Using structured fallback.", flush=True)

        # Step 2: Intelligent Instant Extractive Summary (0ms LLM overhead!)
        paragraphs = [p.strip() for p in raw_extract.split("\n") if len(p.strip()) > 60]
        summary = "\n\n".join(paragraphs[:3]) if paragraphs else raw_extract[:1400]
        if len(summary.split()) > 200:
            words = summary.split()[:180]
            summary = " ".join(words) + "..."

        country = "International"
        state = ""
        locality = []

        # Parse location cleanly from Wikipedia short_desc (e.g. "Beach on Espiritu Santo in Vanuatu" -> Vanuatu)
        if short_desc:
            desc_parts = [p.strip() for p in re.split(r",| in | at ", short_desc)]
            if len(desc_parts) >= 2:
                country = desc_parts[-1]
                state = desc_parts[-2] if len(desc_parts) >= 3 else ""

        # If user explicitly requested a country, respect it
        user_countries = preferences.get("country_names", [])
        if user_countries and user_countries[0].strip():
            country = user_countries[0]

        return {
            "summary": summary,
            "country": country,
            "state": state,
            "locality": locality,
            "place_of_interests": preferences.get("place_of_interests", ["nature"]),
            "vibe": preferences.get("vibe", ["relaxing"]),
            "place_of_activites": preferences.get("place_of_activites", ["sightseeing"])
        }

    def generate_ondemand_summary(self, title: str, raw_extract: str = "", user_context: str = "", preferences: dict = None) -> dict:
        """
        Solution 2: Generates an in-depth ~300-500 word AI travel guide ON-DEMAND for a single destination
        when the user requests it or opens the modal drawer, avoiding multi-minute delays during search.
        """
        if preferences is None:
            preferences = {}

        if not raw_extract:
            wv_res = self._fetch_wikivoyage_destination(title)
            if wv_res:
                raw_extract = wv_res.get("raw_extract", "")
            else:
                wiki_res = self._fetch_wikipedia_entity_destination(title)
                if wiki_res:
                    raw_extract = wiki_res.get("raw_extract", "")

        if self.llm is not None and getattr(self.llm, "llm", None) is not None:
            try:
                print(f"[SearchAgent] Generating on-demand 500-word AI guide for '{title}'...", flush=True)
                user_prompt = build_full_source_summarizer_user_prompt(title, raw_extract, user_context)
                response = self.llm.call_llm(FULL_SOURCE_SUMMARIZER_PROMPT, user_prompt, max_tokens=650)
                raw_json = response["choices"][0]["message"]["content"]
                parsed = json.loads(raw_json)
                summary = parsed.get("summary_500_words") or parsed.get("summary") or ""
                if len(summary.split()) < 50 and raw_extract:
                    summary = f"{summary}\n\n{raw_extract[:1200]}".strip()
                return {
                    "name": title,
                    "summary_500_words": summary,
                    "country": parsed.get("country") or "",
                    "state": parsed.get("state") or "",
                    "locality": parsed.get("locality") or [],
                    "vibe": parsed.get("vibe") or [],
                    "place_of_activites": parsed.get("place_of_activites") or [],
                    "is_ai_generated": True
                }
            except Exception as e:
                print(f"[SearchAgent] On-demand summary LLM notice: {e}", flush=True)

        paragraphs = [p.strip() for p in raw_extract.split("\n") if len(p.strip()) > 60]
        summary = "\n\n".join(paragraphs[:4]) if paragraphs else raw_extract[:1600]
        return {
            "name": title,
            "summary_500_words": summary,
            "country": "International",
            "state": "",
            "locality": [],
            "vibe": ["scenic", "relaxing"],
            "place_of_activites": ["sightseeing"],
            "is_ai_generated": False
        }

    SEMANTIC_EXPANSIONS = {
        "mountain": ["mountain", "mount", "peak", "summit", "range", "hill", "ridge", "gorge", "alp", "cliff", "highland"],
        "beach": ["beach", "coast", "coastal", "shore", "sea", "ocean", "cove", "bay", "sand", "seaside"],
        "lake": ["lake", "reservoir", "loch", "lagoon", "waterfront", "waterway"],
        "forest": ["forest", "woods", "woodland", "jungle", "rainforest", "pine", "timber", "grove"],
        "island": ["island", "isle", "atoll", "archipelago", "cay"],
        "desert": ["desert", "dunes", "oasis", "canyon", "arid"],
        "river": ["river", "stream", "creek", "waterfall", "falls", "rapids"],
        "castle": ["castle", "fortress", "palace", "chateau", "citadel", "ruins"],
        "temple": ["temple", "shrine", "monastery", "pagoda", "cathedral", "church", "mosque"],
        "wine": ["wine", "vineyard", "winery", "cellar", "viticulture"],
        "hiking": ["hike", "hiking", "trail", "trek", "walk", "climb", "path", "trekking"],
        "camping": ["camp", "camping", "campsite", "tent", "bivouac"],
        "sunset": ["sunset", "sunrise", "dusk", "dawn", "twilight", "golden hour"],
        "diving": ["dive", "diving", "scuba", "snorkel", "snorkeling", "reef"],
        "surfing": ["surf", "surfing", "wave", "swell", "break"],
        "stargazing": ["stargazing", "dark sky", "astronomy", "observatory", "night sky"],
        "skiing": ["ski", "skiing", "snowboard", "snowboarding", "snow", "resort"],
        "wildlife": ["wildlife", "safari", "animals", "birds", "sanctuary", "fauna"]
    }

    def _match_concept(self, concept: str, text: str) -> bool:
        """
        Dynamically matches any concept or its synonyms against text.
        """
        concept = concept.lower().strip()
        if not concept:
            return False
        if concept in text:
            return True
        expansions = self.SEMANTIC_EXPANSIONS.get(concept, [])
        if any(syn in text for syn in expansions):
            return True
        for key, syns in self.SEMANTIC_EXPANSIONS.items():
            if key in concept or any(s in concept for s in syns):
                if any(syn in text for syn in syns):
                    return True
        return False

    # =========================================================================
    # 4. HEURISTIC CONFIDENCE EVALUATION (Dynamic Conjunction & Penalties)
    # =========================================================================

    def evaluate_confidence(self, title: str, summary: str, preferences: dict) -> float:
        """
        Dynamic heuristic scoring for ANY combination of travel interests and activities:
        1. Hard drop for corporate keywords/non-destinations.
        2. Location constraint & strict mismatch penalty.
        3. Dynamic Joint Conjunction & Multi-Feature Evaluation (scales with any combination:
           e.g. mountain+beach, lake+forest, desert+camping, castle+wine, etc.).
        4. Activity & Vibe matching.
        """
        text = f"{title} {summary}".lower()

        # Hard drop for corporate keywords
        if any(w in text for w in ["uk travel agency", "travel retailer", "online retailer", "retail group plc", "registered in england", "retailer specialising"]):
            return 0.0

        countries = [c.lower() for c in preferences.get("country_names", []) if c.strip()]
        cities = [c.lower() for c in preferences.get("city_names", []) if c.strip()]
        interests = [i.lower() for i in preferences.get("place_of_interests", []) if i.strip()]
        vibes = [v.lower() for v in preferences.get("vibe", []) if v.strip()]
        activities = [a.lower() for a in preferences.get("place_of_activites", []) if a.strip()]
        specifics = [s.lower() for s in preferences.get("specific_named_places", []) if s.strip()]
        raw_input = preferences.get("raw_user_input", "").lower()

        # 1. Location Constraint
        location_score = 0.0
        if countries or cities:
            matched_loc = False
            if countries and any(c in text for c in countries):
                matched_loc = True
            if cities and any(ct in text for ct in cities):
                matched_loc = True
            if not matched_loc:
                return 0.15  # Strict location penalty
            location_score = 0.30
        else:
            location_score = 0.25

        # 2. Dynamic Joint Conjunction across ALL requested interests
        all_interests = list(interests)
        for key in self.SEMANTIC_EXPANSIONS:
            if key in raw_input and key not in all_interests:
                all_interests.append(key)

        if len(all_interests) >= 2:
            matched_count = sum(1 for i in all_interests if self._match_concept(i, text))
            ratio = matched_count / len(all_interests)
            if ratio == 1.0:
                co_occur_score = 0.40  # Perfect joint fulfillment of all requested features!
            elif ratio >= 0.5:
                # Partial match: penalize missing elements proportionally
                co_occur_score = round(0.40 * (ratio ** 1.5), 2)
            else:
                co_occur_score = 0.05
        elif len(all_interests) == 1:
            co_occur_score = 0.35 if self._match_concept(all_interests[0], text) else 0.10
        else:
            co_occur_score = 0.25

        # 3. Dynamic Activity & Vibe Bonus
        act_score = 0.0
        all_activities = list(activities)
        for act_term in ["hiking", "camping", "sunset", "diving", "surfing", "wine", "stargazing", "skiing"]:
            if act_term in raw_input and act_term not in all_activities:
                all_activities.append(act_term)

        if all_activities:
            matched_acts = sum(1 for a in all_activities if self._match_concept(a, text))
            act_score += min(0.20, (matched_acts / len(all_activities)) * 0.20)
        else:
            act_score += 0.10

        if vibes:
            matched_vibes = sum(1 for v in vibes if self._match_concept(v, text))
            act_score += min(0.10, (matched_vibes / len(vibes)) * 0.10)
        else:
            act_score += 0.05

        # Specific place bonus (e.g. Bondi, Fuji, Amalfi)
        if specifics and any(s in text for s in specifics):
            act_score = min(0.30, act_score + 0.10)

        total = location_score + co_occur_score + act_score
        return min(1.0, max(0.10, round(total, 2)))

    # =========================================================================
    # 5. ONLINE SEARCH MAIN WORKFLOW
    # =========================================================================

    def search_online(self, preferences: dict, max_results: int = 10) -> list:
        """
        Performs hybrid online retrieval:
        1. Formulates dynamic joint semantic queries pairing any co-occurring requirements together.
        2. Retrieves curated travel destinations from Wikivoyage first.
        3. Retrieves filtered destinations from Wikipedia second.
        4. Evaluates candidates dynamically and attaches raw extracts for on-demand AI summarization.
        """
        country = ", ".join([c for c in preferences.get("country_names", []) if c.strip()])
        city = ", ".join([c for c in preferences.get("city_names", []) if c.strip()])
        interests = [i for i in preferences.get("place_of_interests", []) if i.strip()]
        activities = [a for a in preferences.get("place_of_activites", []) if a.strip()]
        raw_input = preferences.get("raw_user_input", "")

        loc_str = f"in {city} {country}".strip() if (city or country) else ""

        # Dynamic Compound Query Construction for ANY combination of interests & activities
        queries = []

        # If user has multiple interests (e.g. mountain + beach, lake + forest, desert + canyon)
        if len(interests) >= 2:
            for i in range(len(interests) - 1):
                int1, int2 = interests[i], interests[i + 1]
                queries.append(f"{int1} near {int2} {loc_str}".strip())
                queries.append(f"{int1} and {int2} travel destinations {loc_str}".strip())
                queries.append(f"scenic {int1} {int2} {loc_str}".strip())
            if activities:
                queries.append(f"{interests[0]} {activities[0]} {loc_str}".strip())
        elif len(interests) == 1:
            main_interest = interests[0]
            if activities:
                queries.append(f"{main_interest} with {activities[0]} {loc_str}".strip())
                queries.append(f"best {main_interest} for {activities[0]} {loc_str}".strip())
            queries.append(f"famous {main_interest} {loc_str}".strip())
            queries.append(f"popular {main_interest} travel destinations {loc_str}".strip())
        elif activities:
            queries.append(f"best travel destinations for {' and '.join(activities[:2])} {loc_str}".strip())
            queries.append(f"top {activities[0]} places {loc_str}".strip())
        else:
            queries.append(f"popular travel destinations {loc_str}".strip())

        # If specific places mentioned
        specifics = [s for s in preferences.get("specific_named_places", []) if s.strip()]
        if specifics:
            queries.insert(0, f"{specifics[0]} {loc_str}".strip())

        # Supplement with keywords from raw_user_input if queries are few
        if raw_input and len(queries) < 4:
            clean_raw = " ".join([w for w in raw_input.split() if w.lower() not in ["i", "want", "to", "go", "a", "the", "in", "and", "near", "with", "for", "of", "some", "where", "can"]])
            if clean_raw:
                queries.append(f"{clean_raw} {loc_str}".strip())

        seen_titles = set()
        candidates_raw = []

        # Step 1: Collect titles from Wikivoyage (Curated Travel Guides)
        for q in queries:
            wv_titles = self._search_wikivoyage(q, limit=4)
            for t in wv_titles:
                t_lower = t.lower()
                if t_lower not in seen_titles:
                    seen_titles.add(t_lower)
                    candidates_raw.append(("wikivoyage", t))

        # Step 2: Collect titles from Wikipedia (Verified Entities)
        for q in queries:
            wiki_titles = self._search_wikipedia(q, limit=4)
            for t in wiki_titles:
                t_lower = t.lower()
                if t_lower not in seen_titles:
                    seen_titles.add(t_lower)
                    candidates_raw.append(("wikipedia", t))

        print(f"[SearchAgent] Discovered {len(candidates_raw)} candidate travel destination(s) across Wikivoyage & Wikipedia.", flush=True)

        # Step 3: Fetch candidate details in parallel
        fetched_destinations = []
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_map = {}
            for source, title in candidates_raw[:16]:
                if source == "wikivoyage":
                    future = executor.submit(self._fetch_wikivoyage_destination, title)
                else:
                    future = executor.submit(self._fetch_wikipedia_entity_destination, title)
                future_map[future] = title

            for future in as_completed(future_map):
                try:
                    res = future.result()
                    if res:
                        fetched_destinations.append(res)
                except Exception:
                    pass

        # Step 4: Pre-score and filter candidates heuristically
        promising = []
        for dest in fetched_destinations:
            h_score = self.evaluate_confidence(dest["name"], dest["raw_extract"][:1500], preferences)
            dest["heuristic_score"] = h_score
            if h_score >= 0.40:
                promising.append(dest)

        promising.sort(key=lambda x: x["heuristic_score"], reverse=True)
        top_destinations = promising[:max_results]

        # Step 5: Fast Extractive Summaries & Precise Metadata (0ms LLM overhead during search!)
        final_results = []
        for dest in top_destinations:
            meta = self.summarize_and_extract_metadata(
                dest["name"],
                dest["raw_extract"],
                preferences,
                short_desc=dest.get("short_desc", ""),
                use_llm=False  # Solution 2: fast search response; in-depth 500-word summary generated on-demand
            )
            if not meta:
                continue

            score = dest["heuristic_score"]
            final_results.append({
                "name": dest["name"],
                "text_content": meta["summary"],
                "raw_extract": dest["raw_extract"][:4000],  # Saved for on-demand 500-word AI guide!
                "latitude": dest["latitude"],
                "longitude": dest["longitude"],
                "place_of_interests": meta["place_of_interests"],
                "vibe": meta["vibe"],
                "place_of_activites": meta["place_of_activites"],
                "country": meta["country"],
                "state": meta["state"],
                "locality": meta["locality"],
                "image_url": dest["image_url"],
                "confidence": f"{int(score * 100)}%",
                "confidence_score": score,
                "source_url": dest["source_url"],
                "source_type": dest["source_type"]
            })

            clean_name = self._clean_ascii(dest["name"])
            clean_country = self._clean_ascii(meta["country"])
            print(f"[SearchAgent] Enriched Destination '{clean_name}' ({clean_country}) -> Score: {int(score*100)}%", flush=True)

        return final_results
