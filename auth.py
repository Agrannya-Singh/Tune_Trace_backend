import os
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin
from firebase_admin import auth as firebase_auth, credentials

# Initialize Firebase Admin once
# Check if running in a cloud environment where ADC (Application Default Credentials) is available
# Or if FIREBASE_CREDENTIALS path is provided in ENV
try:
    if not firebase_admin._apps:
        # Prevent committing the physical service account JSON by supporting it directly via ENV string in production
        # Base64 is the safest way to pass JSON in Azure/Docker without escaping issues
        firebase_b64 = os.environ.get("FIREBASE_SERVICE_ACCOUNT_BASE64")
        if firebase_b64:
            import json
            import base64
            cred_dict = json.loads(base64.b64decode(firebase_b64).decode('utf-8'))
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
        else:
            # Fallback to GOOGLE_APPLICATION_CREDENTIALS file
            firebase_admin.initialize_app()
except Exception as e:
    import logging
    logging.warning(f"Failed to initialize Firebase Admin: {e}")

security = HTTPBearer(auto_error=False)

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Verify Firebase JWT and return the authenticated user's claims."""
    if not credentials:
        # If no credentials, we allow it to be None and let the route handle if it's required or not
        return None
        
    try:
        token = credentials.credentials
        decoded = firebase_auth.verify_id_token(token)
        return {
            "uid": decoded["uid"],
            "email": decoded.get("email"),
            "name": decoded.get("name"),
        }
    except Exception as e:
        import logging
        logging.error(f"Error verifying Firebase token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired Firebase token",
        )
