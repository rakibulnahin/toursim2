import os
import json
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from db_setup import DB_PATH, COLLECTION_NAME, setup_chroma_db

class DatabaseAgent:
    """
    Database Agent:
    - Connects to Chroma DB via LlamaIndex.
    - Queries database for places based on user preferences.
    - Verifies database return format.
    """
    def __init__(self):
        self.db_path = DB_PATH
        self.collection_name = COLLECTION_NAME
        self._init_index()

    def _init_index(self):
        db = chromadb.PersistentClient(path=self.db_path)
        collections = [col.name for col in db.list_collections()]
        
        if self.collection_name not in collections or db.get_collection(self.collection_name).count() == 0:
            print("[DatabaseAgent] Collection empty or missing. Triggering database setup...")
            self.index = setup_chroma_db()
        else:
            chroma_collection = db.get_collection(self.collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            try:
                embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", model_kwargs={"local_files_only": True})
            except Exception:
                embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                embed_model=embed_model
            )
            print(f"[DatabaseAgent] Connected to Chroma collection '{self.collection_name}' with {chroma_collection.count()} entries.")

    def search_database(self, preferences: dict, top_k: int = 5) -> list:
        """
        Queries Chroma DB using LlamaIndex retriever and returns matched places.
        """
        # Construct query query string from preferences dict
        query_parts = []
        if preferences.get("place_of_interests"):
            query_parts.append(f"Interests: {', '.join(preferences['place_of_interests'])}")
        if preferences.get("vibe"):
            query_parts.append(f"Vibe: {', '.join(preferences['vibe'])}")
        if preferences.get("place_of_activites"):
            query_parts.append(f"Activities: {', '.join(preferences['place_of_activites'])}")
        if preferences.get("country_names"):
            query_parts.append(f"Country: {', '.join(preferences['country_names'])}")
        if preferences.get("city_names"):
            query_parts.append(f"City: {', '.join(preferences['city_names'])}")
        if preferences.get("specific_named_places"):
            query_parts.append(f"Places: {', '.join(preferences['specific_named_places'])}")

        fallback_query = preferences.get("raw_user_input", "popular tourist place")
        query_str = " ".join(query_parts) if query_parts else fallback_query
        print(f"[DatabaseAgent] Executing vector query: '{query_str}'", flush=True)

        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query_str)

        results = []
        for node in nodes:
            meta = node.metadata
            raw_json_str = meta.get("raw_json", "{}")
            try:
                place_dict = json.loads(raw_json_str)
            except Exception:
                place_dict = {
                    "name": meta.get("name"),
                    "text_content": node.get_content(),
                    "country": meta.get("country"),
                    "state": meta.get("state"),
                    "locality": meta.get("locality", "").split(", "),
                    "latitude": meta.get("latitude"),
                    "longitude": meta.get("longitude"),
                    "image_url": meta.get("image_url")
                }
            place_dict["score"] = float(node.score) if hasattr(node, "score") and node.score is not None else 0.0
            
            # Verify database format before appending
            if self.verify_database_format(place_dict):
                results.append(place_dict)

        print(f"[DatabaseAgent] Found {len(results)} valid format matches from DB.")
        return results

    def verify_database_format(self, place_data: dict) -> bool:
        """
        Verifies that place data adheres to expected database format schema.
        """
        required_fields = ["name", "text_content", "country"]
        for field in required_fields:
            if field not in place_data or not place_data[field]:
                print(f"[DatabaseAgent] Verification failed: missing field '{field}' in {place_data}")
                return False
        return True

    def insert_places(self, new_places: list) -> int:
        """
        Inserts and indexes newly discovered places from SearchAgent into ChromaDB
        and appends them to chroma_sample.json for persistent future use.
        """
        if not new_places:
            return 0

        # Load existing names from chroma_sample.json to avoid duplicates
        sample_path = "chroma_sample.json"
        existing_places = []
        if os.path.exists(sample_path):
            try:
                with open(sample_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    existing_places = data if isinstance(data, list) else [data]
            except Exception:
                existing_places = []

        existing_names = {p.get("name", "").strip().lower() for p in existing_places if isinstance(p, dict)}

        from llama_index.core import Document
        added_count = 0

        for place in new_places:
            name = place.get("name", "").strip()
            if not name or name.lower() in existing_names:
                continue

            vibe_str = ", ".join(place.get("vibe", [])) if isinstance(place.get("vibe"), list) else str(place.get("vibe", ""))
            interests_str = ", ".join(place.get("place_of_interests", [])) if isinstance(place.get("place_of_interests"), list) else str(place.get("place_of_interests", ""))
            activities_str = ", ".join(place.get("place_of_activites", [])) if isinstance(place.get("place_of_activites"), list) else str(place.get("place_of_activites", ""))
            locality_str = ", ".join(place.get("locality", [])) if isinstance(place.get("locality"), list) else str(place.get("locality", ""))

            text_payload = (
                f"Place Name: {name}\n"
                f"Description: {place.get('text_content', '')}\n"
                f"Country: {place.get('country', '')}, State: {place.get('state', '')}, Locality: {locality_str}\n"
                f"Vibe: {vibe_str}\n"
                f"Points of Interest: {interests_str}\n"
                f"Activities: {activities_str}"
            )

            # Clean place object for serialization
            clean_place = {
                "name": name,
                "text_content": place.get("text_content", ""),
                "latitude": float(place.get("latitude", 0.0)),
                "longitude": float(place.get("longitude", 0.0)),
                "place_of_interests": place.get("place_of_interests", []),
                "vibe": place.get("vibe", []),
                "place_of_activites": place.get("place_of_activites", []),
                "country": place.get("country", "International"),
                "state": place.get("state", ""),
                "locality": place.get("locality", []),
                "image_url": place.get("image_url", ""),
                "source_url": place.get("source_url", "")
            }

            doc = Document(
                text=text_payload,
                metadata={
                    "name": name,
                    "country": clean_place["country"],
                    "state": clean_place["state"],
                    "locality": locality_str,
                    "vibe": vibe_str,
                    "place_of_interests": interests_str,
                    "place_of_activites": activities_str,
                    "latitude": clean_place["latitude"],
                    "longitude": clean_place["longitude"],
                    "image_url": clean_place["image_url"],
                    "raw_json": json.dumps(clean_place)
                }
            )

            try:
                self.index.insert(doc)
                existing_places.append(clean_place)
                existing_names.add(name.lower())
                added_count += 1
            except Exception as e:
                print(f"[DatabaseAgent] Notice: Could not insert '{name}' into Chroma index: {e}")

        # Persist updated place list back to chroma_sample.json
        if added_count > 0:
            try:
                with open(sample_path, "w", encoding="utf-8") as f:
                    json.dump(existing_places, f, indent=2)
                print(f"[DatabaseAgent] Successfully saved {added_count} new destination(s) into ChromaDB & '{sample_path}' for future queries!", flush=True)
            except Exception as e:
                print(f"[DatabaseAgent] Notice: Could not update '{sample_path}': {e}")

        return added_count

if __name__ == "__main__":
    agent = DatabaseAgent()
    sample_prefs = {
        "vibe": ["adventure", "nature"],
        "place_of_interests": ["mountain", "national park"],
        "country_names": ["Australia"]
    }
    matches = agent.search_database(sample_prefs)
    print("Database Query Results:")
    print(json.dumps(matches, indent=2))
