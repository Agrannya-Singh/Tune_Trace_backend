# Deployment Guide

This guide covers deploying the TuneTrace Backend API to production with OAuth support.

---

## Prerequisites

- Render account (or similar hosting)
- YouTube Data API v3 key
- PostgreSQL database (Render provides free tier)
- Redis instance (optional but recommended)
- Git repository

---

## Environment Setup

### Required Environment Variables

```bash
# YouTube API Key (Required)
YOUTUBE_API_KEY=AIzaSy...

# PostgreSQL Database (Production)
POSTGRES_DATABASE_URL=postgresql://user:password@host:port/dbname

# Redis Cache (Optional - Recommended for production)
REDIS_URL=redis://default:password@host:port
REDIS_TTL_SECONDS=3600

# SQLite Fallback (Development only)
SQLITE_DATABASE_URL=sqlite:///app.db

# Database Read Preference
DB_READ_PREFERENCE=postgres
```

### Getting API Keys

#### YouTube Data API v3
1. Go to [Google Cloud Console](https://console.developers.google.com/)
2. Create a new project or select existing
3. Enable "YouTube Data API v3"
4. Create credentials → API Key
5. Restrict key to YouTube Data API v3 (recommended)
6. Copy the API key

---

## Render Deployment

### Step 1: Create Web Service

1. Log in to [Render](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `tune-trace-backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free or Starter (depending on traffic)

### Step 2: Add PostgreSQL Database

1. In Render dashboard, click "New +" → "PostgreSQL"
2. Configure:
   - **Name**: `tunetrace-db`
   - **Database**: `tunetrace`
   - **User**: Auto-generated
   - **Region**: Same as web service
3. Note: Render automatically sets `DATABASE_URL` environment variable
4. Manually add to web service:
   ```
   POSTGRES_DATABASE_URL=${DATABASE_URL}
   ```

### Step 3: Add Redis (Optional)

1. In Render dashboard, click "New +" → "Redis"
2. Configure:
   - **Name**: `tunetrace-cache`
   - **Region**: Same as web service
   - **Plan**: Free tier
3. Copy the internal Redis URL
4. Add to web service environment:
   ```
   REDIS_URL=redis://default:password@host:port
   ```

### Step 4: Set Environment Variables

In your web service settings → Environment:

```bash
YOUTUBE_API_KEY=your_api_key_here
POSTGRES_DATABASE_URL=${DATABASE_URL}
REDIS_URL=redis://...
REDIS_TTL_SECONDS=3600
DB_READ_PREFERENCE=postgres
```

### Step 5: Run Database Migrations

After first deployment:

```bash
# Via Render Shell
alembic upgrade head

# Or connect via SSH and run
psql $DATABASE_URL
# Then manually apply migration SQL from alembic/versions/
```

### Step 6: Verify Deployment

```bash
# Health check
curl https://your-app.onrender.com/health

# Expected response
{"status":"healthy"}
```

---

## OAuth Integration with Frontend

### Frontend Configuration

The backend expects `user_id` to be the user's email from OAuth.

#### Next.js Frontend Setup

**1. Install NextAuth.js**:
```bash
npm install next-auth
```

**2. Configure Google Provider** (`pages/api/auth/[...nextauth].ts`):
```typescript
import NextAuth from "next-auth"
import GoogleProvider from "next-auth/providers/google"

export default NextAuth({
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
  ],
  callbacks: {
    async session({ session, token }) {
      // Session now includes user email
      return session
    },
  },
})
```

**3. Extract User Email** (in your components):
```typescript
import { useSession } from "next-auth/react"

function MyComponent() {
  const { data: session } = useSession()
  
  const getSuggestions = async (songs: string[]) => {
    if (!session?.user?.email) {
      console.error("User not authenticated")
      return
    }
    
    const response = await fetch("https://your-backend.onrender.com/suggestions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: session.user.email,  // ← OAuth email
        songs: songs,
      })
    })
    
    return await response.json()
  }
  
  // ... rest of component
}
```

### Backend User Flow

1. **User Signs In**: Frontend authenticates via Google OAuth
2. **Extract Email**: Frontend gets `session.user.email`
3. **API Request**: Frontend sends `user_id: "user@gmail.com"`
4. **Backend Processing**:
   ```python
   # In main.py POST /suggestions
   user = repo.get_or_create_user(request.user_id)
   # user_id = "user@gmail.com" stored in database
   ```
5. **Personalization**: User's preferences persist across sessions/devices

---

## Database Management

### Running Migrations

**Create a new migration**:
```bash
alembic revision --autogenerate -m "description"
```

**Apply migrations**:
```bash
alembic upgrade head
```

**Rollback migration**:
```bash
alembic downgrade -1
```

### Manual Migration (if Alembic unavailable)

Connect to your PostgreSQL database:
```bash
psql $POSTGRES_DATABASE_URL
```

Run migration SQL:
```sql
-- Add OAuth fields to users table
ALTER TABLE users ADD COLUMN name VARCHAR(255);
ALTER TABLE users ADD COLUMN email VARCHAR(255);
ALTER TABLE users ADD COLUMN updated_at TIMESTAMP DEFAULT NOW();
ALTER TABLE users ALTER COLUMN user_id TYPE VARCHAR(255);
CREATE INDEX idx_users_email ON users(email);
```

### Backup Database

```bash
# Export
pg_dump $POSTGRES_DATABASE_URL > backup.sql

# Restore
psql $POSTGRES_DATABASE_URL < backup.sql
```

---

## Performance Tuning

### Redis Configuration

**For high traffic**:
```bash
REDIS_TTL_SECONDS=1800  # 30 minutes for frequently accessed data
```

**For low traffic**:
```bash
REDIS_TTL_SECONDS=7200  # 2 hours to maximize cache hits
```

### Database Connection Pooling

Already configured in `db.py`:
```python
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True for debugging
    pool_size=10,  # Adjust based on concurrent users
    max_overflow=20
)
```

### YouTube API Quota Management

**Daily Quota**: 10,000 units

**Monitor Usage**:
- Each search: 100 units
- Each video fetch: 1 unit
- Average request: ~101 units

**Optimization**:
- LRU cache reduces repeated searches
- Collaborative filtering reduces YouTube API calls
- Estimated capacity: ~99 full recommendation cycles/day

**If quota exceeded**:
1. Request quota increase from Google
2. Implement request queuing
3. Use multiple API keys (advanced)

---

## Monitoring & Debugging

### Health Checks

**Endpoint**: `GET /health`

**Render Health Check Configuration**:
- Path: `/health`
- Expected Status: 200
- Timeout: 30s

### Logs

**View logs in Render**:
1. Go to your web service
2. Click "Logs" tab
3. Filter by error level

**Log Levels**:
- `INFO`: Normal operations
- `WARNING`: Non-critical issues (e.g., empty queries)
- `ERROR`: YouTube API failures
- `CRITICAL`: Database connection failures

### Common Issues

**Issue**: "Service is not configured"
- **Cause**: Missing `YOUTUBE_API_KEY`
- **Fix**: Set environment variable in Render

**Issue**: "External service is unavailable"
- **Cause**: YouTube API quota exceeded or network issue
- **Fix**: Check quota in Google Cloud Console

**Issue**: "Database connection failed"
- **Cause**: Invalid `POSTGRES_DATABASE_URL`
- **Fix**: Verify database is running and URL is correct

**Issue**: Redis connection errors (non-critical)
- **Cause**: Redis not configured or unreachable
- **Fix**: Service continues without Redis, but with degraded performance

---

## Security Best Practices

### API Key Protection
- ✅ Never commit API keys to Git
- ✅ Use Render's secret environment variables
- ✅ Restrict YouTube API key to specific APIs
- ✅ Rotate keys periodically

### CORS Configuration

**Current** (development):
```python
allow_origins=["*"]  # Allow all origins
```

**Production** (recommended):
```python
allow_origins=[
    "https://your-frontend.vercel.app",
    "https://tunetrace.onrender.com"
]
```

Update in `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tune-trace-rubp.vercel.app"],  # Your frontend URLs
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
```

### Rate Limiting (Future Enhancement)

Consider adding:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/suggestions")
@limiter.limit("10/minute")  # 10 requests per minute per IP
async def post_suggestions(...):
    # ...
```

---

## Testing Deployment

### Test Suite

```bash
# Test health endpoint
curl https://your-backend.onrender.com/health

# Test suggestions endpoint
curl -X POST https://your-backend.onrender.com/suggestions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test@example.com",
    "songs": ["Blinding Lights - The Weeknd", "Shape of You - Ed Sheeran"]
  }'

# Test liked songs endpoint
curl "https://your-backend.onrender.com/liked-songs?user_id=test@example.com"
```

### Expected Responses

**Health**:
```json
{"status": "healthy"}
```

**Suggestions** (first time user - fallback):
```json
{
  "suggestions": [
    {
      "title": "Popular Song Title",
      "artist": "Artist Name",
      "youtube_video_id": "abc123"
    }
  ]
}
```

**Liked Songs** (new user):
```json
[]
```

---

## Rollback Procedure

If deployment fails:

1. **Revert Code**:
   ```bash
   git revert HEAD
   git push
   ```

2. **Rollback Database** (if migration applied):
   ```bash
   alembic downgrade -1
   ```

3. **Check Logs**:
   - Identify error in Render logs
   - Fix issue locally
   - Test thoroughly
   - Redeploy

---

## Support & Troubleshooting

### Resources
- **API Documentation**: See `API_GUIDE.md`
- **Changelog**: See `CHANGELOG.md`
- **GitHub Issues**: https://github.com/Agrannya-Singh/Tune_Trace_backend/issues

### Contact
- **Email**: singh.agrannya@gmail.com
- **Production URL**: https://song-suggest-microservice.onrender.com

---

**Last Updated**: September 30, 2025  
**Version**: 2.0.0
