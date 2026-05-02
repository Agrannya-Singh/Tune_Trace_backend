

## TuneTrace - Semantic Music Recommendation API

A FastAPI microservice that provides AI-powered music recommendations using **semantic vector search** (`all-MiniLM-L6-v2` + `pgvector`). The engine encodes song metadata into 384-dimensional dense vectors and retrieves the most semantically similar tracks via cosine distance on Supabase/PostgreSQL.

**Production URL**: https://song-suggest-fasapi.azurewebsites.net

---

## Architecture (V3 - Semantic)

```
                         ┌──────────────────────────────┐
                         │   SentenceTransformer Model   │
                         │    all-MiniLM-L6-v2 (384-d)  │
                         └────────────┬─────────────────┘
                                      │ encode()
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
   POST /suggestions           POST /discover           seed_trending_music.py
   (User History →             (Free Text →              (Trending Songs →
    Profile Vector)             Query Vector)              Store Embeddings)
            │                         │                         │
            └─────────────┬───────────┘                         │
                          ▼                                     ▼
                  ┌──────────────────┐                 ┌──────────────┐
                  │  pgvector <=>    │                 │  Supabase DB │
                  │  Cosine Distance │◄────────────────│  (HNSW Index)│
                  └───────┬──────────┘                 └──────────────┘
                          ▼
                   Top-N Results
```

---

## API Contract

### Endpoints

#### 1. POST /suggestions
**Description**: Get AI-powered song suggestions based on a user's liked songs using **semantic vector search**.

**Request Body** (JSON):
```json
{
  "user_id": "user@example.com",
  "songs": ["Shape of You - Ed Sheeran", "Blinding Lights - The Weeknd"],
  "genre": "Pop"
}
```

**Fields**:
- `user_id` (required): User email or OAuth identifier (max 255 chars)
- `songs` (required): Array of song titles, 1–50 items
- `genre` (optional): Genre for fallback suggestions (max 128 chars)

**Response** (200 OK):
```json
{
  "suggestions": [
    {
      "title": "Billie Eilish - bad guy",
      "artist": "Billie Eilish",
      "youtube_video_id": "kJQP7kiw5Fk"
    }
  ]
}
```

**Errors**:
- 400 (invalid input — exceeds limits)
- 500 (internal server error)
- 503 (YouTube API unavailable)

---

#### 2. POST /discover
**Description**: Semantic discovery - search by mood, song name, or free-text description. No user history or authentication required.

**Request Body** (JSON):
```json
{
  "query": "chill lo-fi vibes for late night studying",
  "limit": 10
}
```

**Fields**:
- `query` (required): Free-text search query - a mood, song name, or vibe (2-500 chars)
- `limit` (optional): Number of results to return, 1–30 (default: 10)

**Response** (200 OK):
```json
{
  "query": "chill lo-fi vibes for late night studying",
  "results": [
    {
      "title": "Lofi Girl - Study Session",
      "artist": "Lofi Girl",
      "youtube_video_id": "jfKfPfyJRdk",
      "genre": "Lo-Fi",
      "score": 0.8742
    },
    {
      "title": "Nujabes - Feather",
      "artist": "Nujabes",
      "youtube_video_id": "M-BWXT3UBns",
      "genre": "Jazz Hip-Hop",
      "score": 0.8234
    }
  ]
}
```

**Errors**:
- 422 (validation error - query too short/long)
- 500 (internal server error)

---

#### 3. GET /liked-songs
**Description**: Returns the list of songs a user has previously liked.

**Query Parameters**:
- `user_id` (required): User email or OAuth identifier

**Example**: `GET /liked-songs?user_id=user@example.com`

**Response** (200 OK):
```json
[
  {
    "video_id": "dQw4w9WgXcQ",
    "title": "Rick Astley - Never Gonna Give You Up",
    "artist": "Official Rick Astley",
    "created_at": "2025-09-30T14:23:45.123456"
  }
]
```

**Errors**:
- 500 (failed to retrieve liked songs)

---

#### 4. GET /health
**Description**: Health check endpoint to confirm service is running.

**Response** (200 OK):
```json
{
  "status": "healthy"
}
```

---

## Recommendation Algorithm (Version 3.0 - Semantic)

**Goal**: Provide high-quality, personalized song recommendations using dense vector semantic search.

### Pipeline

| Step | Description |
|:---|:---|
| 1. **Encoding** | Each song's title, artist, genre, and tags are formatted into a natural-language text context and encoded into a 384-d vector by `all-MiniLM-L6-v2`. |
| 2. **Profile Vector** | A recency-decay-weighted average of the user's liked song vectors produces a single profile vector. Recent likes carry more weight via exponential decay (`e^(-t)`). |
| 3. **pgvector Retrieval** | The profile vector is queried against the HNSW-indexed `embedding` column using `<=>` cosine distance, returning the top candidates in O(log N) time. |
| 4. **Exclusion** | Previously liked and previously recommended songs (tracked in Redis, 24h TTL) are excluded at the SQL level. |
| 5. **Diversity** | 60% strict similarity + 40% random sampling from remaining candidates prevents filter bubbles. |
| 6. **Fallback** | If zero vectorized candidates are found, the system falls back to genre-based YouTube trending search. |

### Discovery Mode (`/discover`)

| Step | Description |
|:---|:---|
| 1. **Encoding** | Raw user text (mood / song name / description) is directly encoded into a 384-d query vector. |
| 2. **Retrieval** | pgvector cosine distance search against the full catalog. |
| 3. **Scoring** | Results include a `score` field (0–1) indicating semantic similarity. |

### Performance Optimizations

**Caching Strategy**:
- **Redis Cache**: User preferences with configurable TTL (default: 1 hour)
- **Model Weight Cache**: SentenceTransformer loaded once at startup (~90 MB RAM, ~3s cold start)
- **HuggingFace CI Cache**: GitHub Actions caches `~/.cache/huggingface/` to skip 80 MB re-downloads
- **Background Tasks**: Redis updates happen asynchronously (non-blocking)
- **HNSW Index**: pgvector HNSW index enables sub-millisecond nearest-neighbour lookups

**Latency Targets**:
- Vector encoding: ~5–15ms per query
- pgvector retrieval: <50ms
- Full `/discover` round-trip: <100ms
- Full `/suggestions` round-trip: <500ms (includes YouTube search for new songs)

---

## Database Schema

### Tables

#### song_metadata
```sql
CREATE TABLE song_metadata (
    id SERIAL PRIMARY KEY,
    video_id VARCHAR(64) UNIQUE NOT NULL,
    title VARCHAR(512) NOT NULL,
    artist VARCHAR(256) NOT NULL,
    genre VARCHAR(128),
    tags TEXT,
    enriched VARCHAR(16),             -- NULL=raw, 'V2'=genre-enriched, 'V3'=vectorized
    embedding vector(384),            -- Dense vector from SentenceTransformer
    updated_at TIMESTAMP DEFAULT NOW()
);

-- HNSW index for fast cosine distance search
CREATE INDEX idx_song_embedding_hnsw
    ON song_metadata USING hnsw (embedding vector_cosine_ops);
```

#### users
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    email VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### user_liked_songs
```sql
CREATE TABLE user_liked_songs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    song_id INTEGER REFERENCES song_metadata(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, song_id)
);
```

### Migration (V2 → V3)
```bash
# Enable pgvector extension (Supabase)
# Already enabled via dashboard or:
CREATE EXTENSION IF NOT EXISTS vector;

# Backfill existing songs with embeddings
python v3_janitor.py
```

---

## Frontend Integration

**OAuth User Flow**:
1. Frontend authenticates user via Google OAuth (NextAuth.js)
2. Extract user email from session: `session.user.email`
3. Pass email as `user_id` in API requests

**Example (fetch)**:
```javascript
// Personalized recommendations (requires liked songs)
async function getSuggestions(userId, songs, genre = null) {
  const res = await fetch("https://song-suggest-fasapi.azurewebsites.net/suggestions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, songs, genre })
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return (await res.json()).suggestions;
}

// Semantic discovery (no auth needed)
async function discoverMusic(query, limit = 10) {
  const res = await fetch("https://song-suggest-fasapi.azurewebsites.net/discover", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, limit })
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return (await res.json()).results;
}

// Liked songs
async function getLikedSongs(userId) {
  const res = await fetch(
    `https://song-suggest-fasapi.azurewebsites.net/liked-songs?user_id=${encodeURIComponent(userId)}`
  );
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}
```

---

## Usage Examples

**POST /suggestions** (personalized recommendations):
```bash
curl -X POST \
  https://song-suggest-fasapi.azurewebsites.net/suggestions \
  -H "Content-Type: application/json" \
  -d '{
        "user_id": "demo-user",
        "songs": ["Blinding Lights", "Shape of You"]
      }'
```

**POST /discover** (mood / free-text search):
```bash
curl -X POST \
  https://song-suggest-fasapi.azurewebsites.net/discover \
  -H "Content-Type: application/json" \
  -d '{
        "query": "sad acoustic songs for a rainy day",
        "limit": 5
      }'
```

**GET /liked-songs**:
```bash
curl "https://song-suggest-fasapi.azurewebsites.net/liked-songs?user_id=demo-user"
```

**GET /health**:
```bash
curl "https://song-suggest-fasapi.azurewebsites.net/health"
```

---

## Configuration

Environment variables:
- `YOUTUBE_API_KEY`: Required.
- `POSTGRES_DATABASE_URL`: Required for production (Supabase connection string).
- `REDIS_URL`: Optional. Redis URL for caching (free tier supported).
- `REDIS_TTL_SECONDS`: Optional. Default `3600`.

Start command (Azure/Local):
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Dependencies:
- See `requirements.txt`. Includes `sentence-transformers`, `pgvector`, and `SQLAlchemy`.

CORS:
- CORS is set to allow all origins by default for ease of integration. Restrict in production as needed.

---

## Health Check
```
GET https://song-suggest-fasapi.azurewebsites.net/health
Response: { "status": "healthy" }
```

---

## Testing (Bruno Collection)

For contributors and developers, this repository includes a **Bruno** collection for testing the API.

1.  **Install Bruno**: [https://www.usebruno.com/](https://www.usebruno.com/)
2.  **Open Collection**: In Bruno, click "Open Collection" and select the `TuneTrace-Backend` folder in the root of this repo.
3.  **Select Environment**: Choose "Production" from the environment dropdown (top right).
4.  **Run Requests**:
    *   **Health**: Check if API is running.
    *   **Get Suggestions**: Trigger the semantic recommendation engine.
    *   **Discover Music**: Search by mood or song name.
    *   **Get Liked Songs**: Verify database/cache persistence.
