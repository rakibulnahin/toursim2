

import os
import json
import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

DB_PATH = "./chroma_db"
COLLECTION_NAME = "tourism_places"

def load_sample_data(filepath="chroma_sample.json"):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return []
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = [data]
    
    # Additional sample entries if needed for testing variety
    additional_sample = [
        {
            "name": "Bondi Beach",
            "text_content": "Bondi Beach is a popular beach and the name of the surrounding suburb in Sydney, New South Wales, Australia. Known for iconic surfing, coastal walks, and vibrant beach houses.",
            "latitude": -33.8915,
            "longitude": 151.2767,
            "place_of_interests": ["beach", "ocean", "coastal walk"],
            "vibe": ["relaxing", "vacation", "surf"],
            "place_of_activites": ["rent a beach house", "swimming", "surfing", "eating pizza", "fishing"],
            "country": "Australia",
            "state": "New South Wales",
            "locality": ["Sydney", "Bondi"],
            "image_url": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e"
        },
        {
            "name": "Westfield Sydney & Opera House Precinct",
            "text_content": "Central Sydney shopping district featuring Westfield shopping mall, luxury dining, La Spiazza restaurant, and proximity to Sydney Opera House.",
            "latitude": -33.8688,
            "longitude": 151.2093,
            "place_of_interests": ["shopping", "landmark", "food", "harbour"],
            "vibe": ["city", "shopping", "foodie"],
            "place_of_activites": ["shopping", "eating pasta", "eating pizza", "sightseeing"],
            "country": "Australia",
            "state": "New South Wales",
            "locality": ["Sydney"],
            "image_url": "https://images.unsplash.com/photo-1528072164453-f4e8ef0d475a"
        }
    ]
    
    # Avoid duplicate names if present
    existing_names = {item.get("name") for item in data}
    for item in additional_sample:
        if item.get("name") not in existing_names:
            data.append(item)
            
    return data

def setup_chroma_db():
    print("Setting up Chroma DB with LlamaIndex...", flush=True)
    db = chromadb.PersistentClient(path=DB_PATH)
    try:
        db.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    
    # Prepare documents for LlamaIndex
    raw_places = load_sample_data("chroma_sample.json")
    documents = []
    
    for place in raw_places:
        vibe_str = ", ".join(place.get("vibe", [])) if isinstance(place.get("vibe"), list) else str(place.get("vibe", ""))
        interests_str = ", ".join(place.get("place_of_interests", [])) if isinstance(place.get("place_of_interests"), list) else str(place.get("place_of_interests", ""))
        activities_str = ", ".join(place.get("place_of_activites", [])) if isinstance(place.get("place_of_activites"), list) else str(place.get("place_of_activites", ""))
        locality_str = ", ".join(place.get("locality", [])) if isinstance(place.get("locality"), list) else str(place.get("locality", ""))
        
        text_payload = (
            f"Place Name: {place.get('name', '')}\n"
            f"Description: {place.get('text_content', '')}\n"
            f"Country: {place.get('country', '')}, State: {place.get('state', '')}, Locality: {locality_str}\n"
            f"Vibe: {vibe_str}\n"
            f"Points of Interest: {interests_str}\n"
            f"Activities: {activities_str}"
        )
        
        doc = Document(
            text=text_payload,
            metadata={
                "name": place.get("name", ""),
                "country": place.get("country", ""),
                "state": place.get("state", ""),
                "locality": locality_str,
                "vibe": vibe_str,
                "place_of_interests": interests_str,
                "place_of_activites": activities_str,
                "latitude": float(place.get("latitude", 0.0)),
                "longitude": float(place.get("longitude", 0.0)),
                "image_url": place.get("image_url", ""),
                "raw_json": json.dumps(place)
            }
        )
        documents.append(doc)
        
    try:
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", model_kwargs={"local_files_only": True})
    except Exception:
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    print(f"Successfully indexed {len(documents)} places into Chroma DB at '{DB_PATH}'.", flush=True)
    return index

if __name__ == "__main__":
    setup_chroma_db()
