import os
import time
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
if not TMDB_API_KEY:
    raise ValueError("TMDB_API_KEY is missing in your environment.")

DB_PATH = "./db/movie_db"
COLLECTION_NAME = "movie_detective"

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

os.makedirs("./db", exist_ok=True)

client = chromadb.PersistentClient(path=DB_PATH)

try:
    collection = client.get_collection(name=COLLECTION_NAME)
except Exception:
    collection = client.create_collection(name=COLLECTION_NAME)

BASE_URL = "https://api.themoviedb.org/3"


def safe_get(url: str, params: dict | None = None, retries: int = 3):
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=20)
            if response.status_code == 429:
                time.sleep(2 + attempt)
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == retries - 1:
                return None
            time.sleep(1 + attempt)
    return None


def build_movie_document(movie: dict, details: dict, keywords_data: dict) -> tuple[str, dict]:
    title = movie.get("title", "")
    original_title = details.get("original_title", "") or movie.get("original_title", "")
    overview = movie.get("overview", "") or details.get("overview", "")
    release_date = movie.get("release_date", "") or details.get("release_date", "")
    year = release_date[:4] if release_date else "N/A"

    genres = [g.get("name", "") for g in details.get("genres", []) if g.get("name")]
    keywords = [k.get("name", "") for k in keywords_data.get("keywords", []) if k.get("name")]
    tagline = details.get("tagline", "") or ""
    collection_name = ""
    if details.get("belongs_to_collection") and details["belongs_to_collection"].get("name"):
        collection_name = details["belongs_to_collection"]["name"]

    production_countries = [
        c.get("name", "") for c in details.get("production_countries", []) if c.get("name")
    ]
    spoken_languages = [
        l.get("english_name", "") for l in details.get("spoken_languages", []) if l.get("english_name")
    ]

    poster_path = movie.get("poster_path") or details.get("poster_path") or ""
    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else ""

    doc_parts = [
        f"Title: {title}",
        f"Original title: {original_title}" if original_title else "",
        f"Year: {year}" if year != "N/A" else "",
        f"Overview: {overview}" if overview else "",
        f"Tagline: {tagline}" if tagline else "",
        f"Genres: {', '.join(genres)}" if genres else "",
        f"Keywords: {', '.join(keywords)}" if keywords else "",
        f"Collection: {collection_name}" if collection_name else "",
        f"Countries: {', '.join(production_countries)}" if production_countries else "",
        f"Languages: {', '.join(spoken_languages)}" if spoken_languages else "",
    ]

    document = ". ".join([part for part in doc_parts if part]).strip()

    metadata = {
        "title": title,
        "original_title": original_title,
        "year": year,
        "overview": overview,
        "tagline": tagline,
        "genres": ", ".join(genres),
        "keywords": ", ".join(keywords),
        "collection": collection_name,
        "poster": poster_url,
    }

    return document, metadata


def ingest_movies(total_pages: int = 150, reset_collection: bool = False):
    global collection

    if reset_collection:
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass
        collection = client.create_collection(name=COLLECTION_NAME)

    existing_ids = set()
    try:
        existing = collection.get(include=[])
        existing_ids = set(existing.get("ids", []))
    except Exception:
        existing_ids = set()

    added_count = 0
    skipped_existing = 0
    skipped_incomplete = 0

    print(f"🚀 Starting ingestion for about {total_pages * 20} movies...")

    for page in range(1, total_pages + 1):
        popular_url = f"{BASE_URL}/movie/popular"
        popular_data = safe_get(popular_url, params={
            "api_key": TMDB_API_KEY,
            "language": "en-US",
            "page": page
        })

        if not popular_data or "results" not in popular_data:
            print(f"⚠️ Skipping page {page} due to fetch error.")
            continue

        ids_batch = []
        docs_batch = []
        metas_batch = []
        embeds_batch = []

        for movie in popular_data["results"]:
            movie_id = str(movie.get("id", ""))
            if not movie_id:
                continue

            if movie_id in existing_ids:
                skipped_existing += 1
                continue

            details_url = f"{BASE_URL}/movie/{movie_id}"
            keywords_url = f"{BASE_URL}/movie/{movie_id}/keywords"

            details = safe_get(details_url, params={"api_key": TMDB_API_KEY, "language": "en-US"})
            keywords_data = safe_get(keywords_url, params={"api_key": TMDB_API_KEY})

            if not details or not keywords_data:
                skipped_incomplete += 1
                continue

            overview = movie.get("overview", "") or details.get("overview", "")
            if not overview:
                skipped_incomplete += 1
                continue

            document, metadata = build_movie_document(movie, details, keywords_data)

            if not document.strip():
                skipped_incomplete += 1
                continue

            try:
                embedding = model.encode(document).tolist()
            except Exception:
                skipped_incomplete += 1
                continue

            ids_batch.append(movie_id)
            docs_batch.append(document)
            metas_batch.append(metadata)
            embeds_batch.append(embedding)

            existing_ids.add(movie_id)

        if ids_batch:
            try:
                collection.add(
                    ids=ids_batch,
                    documents=docs_batch,
                    metadatas=metas_batch,
                    embeddings=embeds_batch
                )
                added_count += len(ids_batch)
            except Exception as e:
                print(f"❌ Failed to add batch on page {page}: {e}")

        if page % 10 == 0 or page == total_pages:
            print(
                f"✅ Page {page}/{total_pages} processed | "
                f"Added: {added_count} | Existing skipped: {skipped_existing} | "
                f"Incomplete skipped: {skipped_incomplete} | Current total: {collection.count()}"
            )

        time.sleep(0.25)

    print(f"\n🏁 Ingestion complete.")
    print(f"Added new movies: {added_count}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Skipped incomplete/error: {skipped_incomplete}")
    print(f"Total movies in collection: {collection.count()}")


if __name__ == "__main__":
    # Set reset_collection=True only if you want a full rebuild.
    ingest_movies(total_pages=150, reset_collection=False)