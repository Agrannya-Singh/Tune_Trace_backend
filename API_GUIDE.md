# TuneTrace Backend API Guide

## Overview

This document provides a comprehensive guide for the TuneTrace backend microservice. TuneTrace is a music discovery application that provides AI-powered song recommendations based on user preferences using YouTube music content.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Technology Stack](#technology-stack)
3. [API Endpoints](#api-endpoints)
4. [Data Models](#data-models)
5. [Authentication Integration](#authentication-integration)
6. [Environment Configuration](#environment-configuration)
7. [Error Handling](#error-handling)
8. [Performance Considerations](#performance-considerations)

---

## System Architecture

### High-Level Architecture

```
┌─────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   Frontend  │ ◄─────► │   FastAPI        │ ◄─────► │  YouTube API    │
│  (Next.js)  │         │   Microservice   │         │                 │
└─────────────┘         └──────────────────┘         └─────────────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  PostgreSQL      │
                        │  + Redis Cache   │
                        └──────────────────┘
```

### Request Flow

1. **User Likes Songs**: Frontend collects liked songs → `/suggestions` endpoint
2. **AI Processing**: Collaborative filtering + YouTube API → Generate recommendations
3. **Return Results**: Video IDs with metadata → Frontend fetches from YouTube
4. **Caching**: Redis caches user likes for sub-200ms latency

---

## Technology Stack

### Backend
- **Framework**: FastAPI 0.115+
- **Language**: Python 3.11+
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Caching**: Redis (sub-200ms latency)
- **External API**: YouTube Data API v3
- **Deployment**: Render
- **Base URL**: `https://song-suggest-microservice.onrender.com`

---

## API Endpoints

### 1. `POST /suggestions` - Get Personalized Recommendations

**Purpose**: Receives liked songs and returns AI-recommended YouTube video IDs.

#### Request Headers
```http
Content-Type: application/json
Authorization: Bearer {access_token}  // Optional: Google OAuth token
```

#### Request Body
```json
{
  "user_id": "user@example.com",
  "songs": [
    "Rick Astley - Never Gonna Give You Up - Official Rick Astley",
    "PSY - GANGNAM STYLE - officialpsy",
    "The Weeknd - Blinding Lights - TheWeekndVEVO"
  ],
  "genre": "Pop"  // Optional: for fallback suggestions
}
```

**Field Descriptions**:
- `user_id` (required): User email or unique identifier from OAuth
- `songs` (required): Array of song title strings (format: "Title - Artist")
- `genre` (optional): Genre for fallback suggestions if collaborative filtering fails

#### Response Format

**Success Response** (200 OK)
```json
{
  "suggestions": [
    {
      "title": "Billie Eilish - bad guy",
      "artist": "Billie Eilish",
      "youtube_video_id": "kJQP7kiw5Fk"
    },
    {
      "title": "The Weeknd - Starboy",
      "artist": "TheWeekndVEVO",
      "youtube_video_id": "fHI8X4OXluQ"
    }
  ]
}
```

**Error Response** (503 Service Unavailable)
```json
{
  "detail": "External service is unavailable."
}
```

**Error Response** (500 Internal Server Error)
```json
{
  "detail": "An internal server error occurred."
}
```

#### Implementation Details

**Recommendation Engine**:
1. **User Identification**: Get or create user by email/ID
2. **Song Resolution**: Search YouTube API for each song title → Get video metadata
3. **Database Persistence**: Store liked songs in PostgreSQL
4. **Collaborative Filtering**: Find users with similar taste (≥2 songs in common)
5. **Recommendations**: Return songs liked by similar users but not by current user
6. **Fallback**: If no collaborative suggestions, fetch popular songs by genre from YouTube
7. **Caching**: Update Redis cache in background task (non-blocking)

**Performance Optimizations**:
- LRU cache (512 entries) for YouTube search results
- Background tasks for Redis updates (response not blocked)
- Batch database operations
- Indexed queries on user_id and song_id

---

### 2. `GET /liked-songs` - Retrieve User's Liked Songs

**Purpose**: Returns the list of songs a user has previously liked.

#### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_id` | string | Yes | User email or unique identifier from OAuth |

#### Request Example

```http
GET /liked-songs?user_id=user@example.com
```

#### Response Format

**Success Response** (200 OK)
```json
[
  {
    "video_id": "dQw4w9WgXcQ",
    "title": "Rick Astley - Never Gonna Give You Up",
    "artist": "Official Rick Astley",
    "created_at": "2025-07-30T14:23:45.123456"
  },
  {
    "video_id": "9bZkp7q19f0",
    "title": "PSY - GANGNAM STYLE",
    "artist": "officialpsy",
    "created_at": "2025-07-30T14:25:12.789012"
  }
]
```

**Error Response** (500 Internal Server Error)
```json
{
  "detail": "Failed to retrieve liked songs."
}
```

#### Implementation Details

- Queries PostgreSQL for user's liked songs joined with song metadata
- Returns results ordered by most recently liked (DESC)
- Returns empty array if user not found or has no liked songs
- Includes ISO 8601 formatted timestamp for each like

---

### 3. `GET /health` - Health Check

**Purpose**: Confirms the service is running.

#### Request Example

```http
GET /health
```

#### Response Format

**Success Response** (200 OK)
```json
{
  "status": "healthy"
}
```

---

## Data Models

### Database Schema

#### Users Table
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,  -- Email or OAuth ID
    name VARCHAR(255),                      -- Display name
    email VARCHAR(255),                     -- Email address
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_users_user_id ON users(user_id);
CREATE INDEX idx_users_email ON users(email);
```

#### Song Metadata Table
```sql
CREATE TABLE song_metadata (
    id SERIAL PRIMARY KEY,
    video_id VARCHAR(64) UNIQUE NOT NULL,  -- YouTube video ID
    title VARCHAR(512) NOT NULL,
    artist VARCHAR(256) NOT NULL,           -- YouTube channel title
    genre VARCHAR(128),
    tags TEXT,                              -- Comma-separated
    updated_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_song_video_id ON song_metadata(video_id);
CREATE INDEX idx_song_genre ON song_metadata(genre);
```

#### User Liked Songs Table
```sql
CREATE TABLE user_liked_songs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    song_id INTEGER REFERENCES song_metadata(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, song_id)
);
CREATE INDEX idx_user_likes_user ON user_liked_songs(user_id);
CREATE INDEX idx_user_likes_song ON user_liked_songs(song_id);
```

### Pydantic Models

#### Request Models
```python
class LikedSongsRequest(BaseModel):
    user_id: str  # User email or OAuth ID
    songs: List[str]  # ["Title - Artist", ...]
    genre: Optional[str] = None

class LikedSongResponse(BaseModel):
    video_id: str
    title: str
    artist: str
    created_at: str  # ISO 8601 format
```

#### Response Models
```python
class SongSuggestion(BaseModel):
    title: str
    artist: str
    youtube_video_id: str

class SuggestionResponse(BaseModel):
    suggestions: List[SongSuggestion]
```

---

## Authentication Integration

### OAuth 2.0 Flow

The frontend uses Google OAuth 2.0 via NextAuth.js. The backend receives the user identifier (email) from the frontend.

**Integration Points**:
1. Frontend authenticates user via Google OAuth
2. Frontend extracts user email from session
3. Frontend sends requests with `user_id` set to user email
4. Backend uses email as unique user identifier

**User Identification**:
```python
# In POST /suggestions
user = repo.get_or_create_user(request.user_id)
# user_id is the email from OAuth (e.g., "user@example.com")
```

**Benefits**:
- No need for separate authentication in backend
- User identity persisted across sessions
- Email serves as natural unique identifier
- Supports multi-device access for same user

---

## Environment Configuration

### Required Environment Variables

**File**: `.env`

```bash
# YouTube Data API v3 Key (Required)
YOUTUBE_API_KEY=AIzaSy...

# PostgreSQL Database (Production)
POSTGRES_DATABASE_URL=postgresql://user:password@host:port/dbname

# Redis Cache (Optional but recommended)
REDIS_URL=redis://default:password@host:port
REDIS_TTL_SECONDS=3600

# SQLite Fallback (Development)
SQLITE_DATABASE_URL=sqlite:///app.db

# Database Read Preference
DB_READ_PREFERENCE=postgres
```

### Configuration Priority

1. **PostgreSQL**: Used if `POSTGRES_DATABASE_URL` is set
2. **SQLite**: Fallback for local development
3. **Redis**: Optional caching layer for performance

### Deployment (Render)

**Start Command**:
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

**Environment Setup**:
1. Set `YOUTUBE_API_KEY` as secret environment variable
2. Add PostgreSQL database addon → `POSTGRES_DATABASE_URL` auto-set
3. Add Redis addon (optional) → `REDIS_URL` auto-set
4. Deploy from GitHub repository

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Usage |
|------|---------|-------|
| 200 | OK | Successful response |
| 400 | Bad Request | Invalid request parameters |
| 404 | Not Found | User not found |
| 500 | Internal Server Error | Unexpected error |
| 503 | Service Unavailable | YouTube API or database unavailable |

### Error Response Structure

```json
{
  "detail": "Human-readable error message"
}
```

### Error Handling Pattern

**Backend Implementation**:
```python
try:
    # Process request
    suggestions = suggestion_service.get_suggestions(user, repo, genre)
    return {"suggestions": suggestions}
except requests.RequestException as e:
    logger.error(f"YouTube API error: {e}")
    raise HTTPException(503, "External service is unavailable.")
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    raise HTTPException(500, "An internal server error occurred.")
```

**Logging**:
- All errors logged with full stack trace
- YouTube API errors logged separately
- User context included in logs (user_id)

---

## Performance Considerations

### Caching Strategy

**1. LRU Cache (In-Memory)**:
```python
@lru_cache(maxsize=512)
def _search_youtube_for_song(self, song_name: str):
    # Cache YouTube search results
```

**2. Redis Cache (Distributed)**:
- User liked songs cached with TTL
- Background task updates (non-blocking)
- Sub-200ms latency for cached data

**3. Database Query Optimization**:
- Indexed queries on user_id, song_id, video_id
- Eager loading with `joinedload()` to prevent N+1 queries
- Batch operations for multiple song likes

### Scalability Targets

- **Concurrent Users**: 100+
- **Database Latency**: <200ms (with Redis)
- **API Response Time**: 40% reduction via caching
- **Uptime**: Production-ready via CI/CD

### YouTube API Quota Management

**Daily Quota**: 10,000 units

**Cost per Operation**:
- Search API: 100 units
- Videos API: 1 unit

**Optimization**:
- Cache search results (512 LRU cache)
- Batch video detail requests
- Fallback to database for repeated queries

---

## Testing

### API Testing Examples

**Test 1: Get Suggestions**
```bash
curl -X POST "https://song-suggest-microservice.onrender.com/suggestions" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test@example.com",
    "songs": ["Blinding Lights - The Weeknd", "Shape of You - Ed Sheeran"]
  }'
```

**Test 2: Get Liked Songs**
```bash
curl "https://song-suggest-microservice.onrender.com/liked-songs?user_id=test@example.com"
```

**Test 3: Health Check**
```bash
curl "https://song-suggest-microservice.onrender.com/health"
```

### Expected Results

- **Suggestions**: Returns 5-10 song recommendations
- **Liked Songs**: Returns list of previously liked songs
- **Health**: Returns `{"status": "healthy"}`

---

## Key Features

### Implemented Features ✅

1. ✅ **OAuth Integration**: Supports user emails from Google OAuth
2. ✅ **Collaborative Filtering**: Finds similar users and recommends their liked songs
3. ✅ **Fallback Mechanism**: Uses popular songs by genre when no collaborative data
4. ✅ **Redis Caching**: Background cache updates for performance
5. ✅ **Database Persistence**: PostgreSQL with SQLAlchemy ORM
6. ✅ **Error Handling**: Comprehensive error handling with proper HTTP codes
7. ✅ **GET /liked-songs**: Endpoint to retrieve user's liked songs
8. ✅ **YouTube Integration**: Searches and fetches song metadata
9. ✅ **CORS Support**: Allows cross-origin requests from frontend

### Recommendation Algorithm

**Hybrid Approach**:
1. **Collaborative Filtering** (Primary):
   - Find users who liked ≥2 same songs
   - Recommend songs those users liked
   - Sort by popularity among similar users

2. **Content-Based Fallback** (Secondary):
   - If no collaborative data available
   - Search YouTube for popular songs by genre
   - Return top results

3. **Caching Layer**:
   - LRU cache for YouTube API calls
   - Redis cache for user preferences
   - Database cache for song metadata

---

## Migration Notes

### From Version 1.0 to 2.0

**Changes**:
1. ✅ Added `name` and `email` fields to User model
2. ✅ Added `GET /liked-songs` endpoint
3. ✅ Enhanced collaborative filtering algorithm
4. ✅ Improved user identification (OAuth email support)
5. ✅ Added Redis caching layer
6. ✅ Background task processing for cache updates

**API Compatibility**:
- ✅ No breaking changes to existing endpoints
- ✅ Backward compatible with frontend
- ✅ Additional fields optional

**Database Migration Required**:
```sql
ALTER TABLE users ADD COLUMN name VARCHAR(255);
ALTER TABLE users ADD COLUMN email VARCHAR(255);
ALTER TABLE users ADD COLUMN updated_at TIMESTAMP DEFAULT NOW();
ALTER TABLE users ALTER COLUMN user_id TYPE VARCHAR(255);
CREATE INDEX idx_users_email ON users(email);
```

---

## Contact & Support

**Project Owner**: Agrannya Singh  
**Email**: singh.agrannya@gmail.com  
**Repository**: https://github.com/Agrannya-Singh/Tune_Trace_backend  
**Production URL**: https://song-suggest-microservice.onrender.com

---

**Document Version**: 2.0  
**Last Updated**: September 30, 2025  
**Status**: Production Ready
