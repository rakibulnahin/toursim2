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
