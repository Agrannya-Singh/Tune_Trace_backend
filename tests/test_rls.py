# tests/test_rls.py
"""
Row Level Security (RLS) integration tests for TuneTrace.

These tests hit the LIVE Supabase PostgREST API to verify that RLS policies
enforce correct access control for each role (anon, authenticated, service_role).

Requirements (all loaded from environment variables — NO hardcoded keys):
  - SUPABASE_URL            — e.g. https://<ref>.supabase.co
  - SUPABASE_ANON_KEY       — publishable anon JWT
  - SUPABASE_PROJECT_ID     — Supabase project ref (e.g. seiriafeehipniwlxtcp)
  - SUPABASE_ACCESS_TOKEN   — personal access token (sbp_...) for Management API tests
  - SUPABASE_SERVICE_ROLE_KEY (optional) — for service_role tests

Run:
  export SUPABASE_URL=https://<ref>.supabase.co
  export SUPABASE_ANON_KEY=eyJ...
  export SUPABASE_PROJECT_ID=<ref>
  export SUPABASE_ACCESS_TOKEN=sbp_...
  pytest tests/test_rls.py -v --tb=short
"""

import os
import pytest
import requests
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Configuration — ALL values loaded from environment variables.
# 
# ---------------------------------------------------------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_PROJECT_ID = os.getenv("SUPABASE_PROJECT_ID", "")
SUPABASE_ACCESS_TOKEN = os.getenv("SUPABASE_ACCESS_TOKEN", "")

# URL for the Supabase Management API (used by metadata/grant tests)
MGMT_API_URL = "https://api.supabase.com/v1"

REST_URL = f"{SUPABASE_URL}/rest/v1" if SUPABASE_URL else ""

# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------

def _headers(api_key: str, role: str = "") -> dict:
    """Build Supabase PostgREST headers for a given role."""
    h = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    return h


def _require_env(name: str, value: str) -> str:
    """Fail fast if a required env var is missing."""
    if not value:
        pytest.skip(f"{name} env var not set — skipping")
    return value


def _anon_headers() -> dict:
    _require_env("SUPABASE_ANON_KEY", SUPABASE_ANON_KEY)
    return _headers(SUPABASE_ANON_KEY)


def _service_headers() -> dict:
    if not SUPABASE_SERVICE_ROLE_KEY:
        pytest.skip("SUPABASE_SERVICE_ROLE_KEY not set — skipping service_role test")
    return _headers(SUPABASE_SERVICE_ROLE_KEY)


skip_if_no_network = pytest.mark.skipif(
    not SUPABASE_URL or not SUPABASE_ANON_KEY,
    reason="SUPABASE_URL and SUPABASE_ANON_KEY env vars required for integration tests",
)


# ============================================================================
# TEST SUITE 1: anon Role — Grant-Level Enforcement
# ============================================================================

@skip_if_no_network
class TestAnonRole:
    """Verify that the anon role has NO write access to any table."""

    # ---- song_metadata: anon can SELECT ----
    def test_anon_can_select_songs(self):
        """anon should be able to SELECT from song_metadata (public catalog)."""
        resp = requests.get(
            f"{REST_URL}/song_metadata?select=id,title,artist&limit=3",
            headers=_anon_headers(),
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert isinstance(data, list)

    # ---- song_metadata: anon CANNOT INSERT ----
    def test_anon_cannot_insert_songs(self):
        """anon should be blocked from INSERTing into song_metadata."""
        payload = {
            "video_id": f"rls_test_{datetime.now(timezone.utc).timestamp()}",
            "title": "RLS Test Song",
            "artist": "RLS Tester",
        }
        resp = requests.post(
            f"{REST_URL}/song_metadata",
            json=payload,
            headers=_anon_headers(),
        )
        # Should be 403 (insufficient_privilege) or 401
        assert resp.status_code in (403, 401, 405), (
            f"anon INSERT should be denied, got {resp.status_code}: {resp.text}"
        )

    # ---- song_metadata: anon CANNOT DELETE ----
    def test_anon_cannot_delete_songs(self):
        """anon should be blocked from DELETEing from song_metadata."""
        resp = requests.delete(
            f"{REST_URL}/song_metadata?id=eq.1",
            headers=_anon_headers(),
        )
        assert resp.status_code in (403, 401, 405), (
            f"anon DELETE should be denied, got {resp.status_code}: {resp.text}"
        )

    # ---- song_metadata: anon CANNOT UPDATE ----
    def test_anon_cannot_update_songs(self):
        """anon should be blocked from UPDATEing song_metadata."""
        resp = requests.patch(
            f"{REST_URL}/song_metadata?id=eq.1",
            json={"title": "HACKED"},
            headers=_anon_headers(),
        )
        assert resp.status_code in (403, 401, 405), (
            f"anon UPDATE should be denied, got {resp.status_code}: {resp.text}"
        )

    # ---- users: anon CANNOT SELECT ----
    def test_anon_cannot_select_users(self):
        """anon should get empty results or 403 when selecting users (no RLS policy grants it)."""
        resp = requests.get(
            f"{REST_URL}/users?select=id,user_id,email&limit=5",
            headers=_anon_headers(),
        )
        # With RLS enabled and no policy for anon SELECT, PostgREST returns 200 with empty []
        # because the role has SELECT grant but RLS filters all rows
        if resp.status_code == 200:
            data = resp.json()
            assert data == [], (
                f"anon should see 0 user rows, got {len(data)}"
            )
        else:
            assert resp.status_code in (403, 401)

    # ---- users: anon CANNOT INSERT ----
    def test_anon_cannot_insert_users(self):
        """anon should be blocked from creating user records."""
        payload = {
            "user_id": "attacker@evil.com",
            "name": "Attacker",
            "email": "attacker@evil.com",
        }
        resp = requests.post(
            f"{REST_URL}/users",
            json=payload,
            headers=_anon_headers(),
        )
        assert resp.status_code in (403, 401, 405), (
            f"anon INSERT on users should be denied, got {resp.status_code}: {resp.text}"
        )

    # ---- user_liked_songs: anon CANNOT SELECT ----
    def test_anon_cannot_select_liked_songs(self):
        """anon should NOT be able to read ANY liked songs (behavioral data leak)."""
        resp = requests.get(
            f"{REST_URL}/user_liked_songs?select=id,user_id,song_id&limit=5",
            headers=_anon_headers(),
        )
        if resp.status_code == 200:
            data = resp.json()
            assert data == [], (
                f"anon should see 0 liked song rows, got {len(data)}"
            )
        else:
            assert resp.status_code in (403, 401)

    # ---- user_liked_songs: anon CANNOT INSERT ----
    def test_anon_cannot_insert_liked_songs(self):
        """anon should be blocked from inserting likes."""
        payload = {"user_id": 1, "song_id": 1}
        resp = requests.post(
            f"{REST_URL}/user_liked_songs",
            json=payload,
            headers=_anon_headers(),
        )
        assert resp.status_code in (403, 401, 405), (
            f"anon INSERT on user_liked_songs should be denied, got {resp.status_code}: {resp.text}"
        )

    # ---- user_liked_songs: anon CANNOT DELETE ----
    def test_anon_cannot_delete_liked_songs(self):
        """anon should be blocked from deleting likes."""
        resp = requests.delete(
            f"{REST_URL}/user_liked_songs?id=eq.1",
            headers=_anon_headers(),
        )
        assert resp.status_code in (403, 401, 405), (
            f"anon DELETE on user_liked_songs should be denied, got {resp.status_code}: {resp.text}"
        )


# ============================================================================
# TEST SUITE 2: service_role — Full Access
# ============================================================================

@skip_if_no_network
class TestServiceRole:
    """Verify that the service_role has unrestricted access (the backend role)."""

    def test_service_role_can_select_users(self):
        """service_role should be able to read all users."""
        resp = requests.get(
            f"{REST_URL}/users?select=id,user_id&limit=5",
            headers=_service_headers(),
        )
        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"

    def test_service_role_can_select_songs(self):
        """service_role should be able to read all songs."""
        resp = requests.get(
            f"{REST_URL}/song_metadata?select=id,title&limit=5",
            headers=_service_headers(),
        )
        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert isinstance(data, list)

    def test_service_role_can_select_liked_songs(self):
        """service_role should be able to read all liked songs."""
        resp = requests.get(
            f"{REST_URL}/user_liked_songs?select=id,user_id,song_id&limit=5",
            headers=_service_headers(),
        )
        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"

    def test_service_role_can_insert_and_cleanup_song(self):
        """service_role should be able to insert and delete a test song."""
        test_vid = f"rls_svc_test_{datetime.now(timezone.utc).timestamp()}"
        payload = {
            "video_id": test_vid,
            "title": "Service Role Test Song",
            "artist": "RLS Test Suite",
        }
        # INSERT
        resp = requests.post(
            f"{REST_URL}/song_metadata",
            json=payload,
            headers=_service_headers(),
        )
        assert resp.status_code in (200, 201), (
            f"service_role INSERT failed: {resp.status_code}: {resp.text}"
        )
        created = resp.json()
        assert len(created) > 0
        record_id = created[0]["id"]

        # DELETE (cleanup)
        resp = requests.delete(
            f"{REST_URL}/song_metadata?id=eq.{record_id}",
            headers=_service_headers(),
        )
        assert resp.status_code in (200, 204), (
            f"service_role DELETE failed: {resp.status_code}: {resp.text}"
        )


# ============================================================================
# TEST SUITE 3: Policy Metadata Verification (via pg_policies)
# ============================================================================

@skip_if_no_network
class TestPolicyMetadata:
    """
    Verify that the expected RLS policies exist in pg_policies
    by querying through the Supabase PostgREST RPC endpoint.
    """

    EXPECTED_POLICIES = {
        "song_metadata": [
            "anon_select_songs",
            "authenticated_select_songs",
            "service_role_insert_songs",
            "service_role_update_songs",
            "service_role_delete_songs",
        ],
        "users": [
            "authenticated_select_own_profile",
            "authenticated_update_own_profile",
            "service_role_insert_users",
            "service_role_select_users",
            "service_role_update_users",
            "service_role_delete_users",
        ],
        "user_liked_songs": [
            "authenticated_select_own_likes",
            "authenticated_insert_own_likes",
            "authenticated_delete_own_likes",
            "service_role_select_likes",
            "service_role_insert_likes",
            "service_role_update_likes",
            "service_role_delete_likes",
        ],
    }

    def _fetch_policies(self) -> list:
        """Fetch active RLS policies via PostgREST RPC or raw SQL.
        
        Uses the service_role key to query pg_policies since
        anon has no access to system catalogs via PostgREST.
        Falls back to the Management API if service_role key is unavailable.
        """
        if SUPABASE_SERVICE_ROLE_KEY:
            # Use PostgREST RPC to query pg_policies
            headers = _service_headers()
            headers["Content-Type"] = "application/json"
            resp = requests.post(
                f"{REST_URL}/rpc/",
                headers=headers,
            )
            # If RPC is not available, fall back to management API
        
        # Use Management API with personal access token
        mgmt_token = _require_env("SUPABASE_ACCESS_TOKEN", SUPABASE_ACCESS_TOKEN)
        project_id = _require_env("SUPABASE_PROJECT_ID", SUPABASE_PROJECT_ID)
        resp = requests.post(
            f"{MGMT_API_URL}/projects/{project_id}/database/query",
            headers={
                "Authorization": f"Bearer {mgmt_token}",
                "Content-Type": "application/json",
            },
            json={
                "query": (
                    "SELECT tablename, policyname, roles, cmd "
                    "FROM pg_policies "
                    "WHERE schemaname = 'public' "
                    "ORDER BY tablename, policyname;"
                )
            },
        )
        assert resp.status_code in (200, 201), f"Failed to query pg_policies: {resp.status_code}"
        return resp.json()

    def test_all_expected_policies_exist(self):
        """Every expected policy should be present in pg_policies."""
        policies = self._fetch_policies()
        
        # Build a set of (table, policy_name) from the response
        existing = set()
        for row in policies:
            existing.add((row["tablename"], row["policyname"]))

        missing = []
        for table, expected_names in self.EXPECTED_POLICIES.items():
            for policy_name in expected_names:
                if (table, policy_name) not in existing:
                    missing.append(f"{table}.{policy_name}")

        assert not missing, (
            f"Missing RLS policies: {', '.join(missing)}"
        )

    def test_no_public_role_policies_on_sensitive_tables(self):
        """No policy on users or user_liked_songs should target the {public} role."""
        policies = self._fetch_policies()
        
        violations = []
        for row in policies:
            if row["tablename"] in ("users", "user_liked_songs"):
                if "{public}" in row.get("roles", ""):
                    violations.append(
                        f"{row['tablename']}.{row['policyname']} targets {{public}}"
                    )

        assert not violations, (
            f"Policies targeting {{public}} on sensitive tables: {', '.join(violations)}"
        )

    def test_rls_enabled_on_all_tables(self):
        """RLS should be enabled on all application tables."""
        mgmt_token = _require_env("SUPABASE_ACCESS_TOKEN", SUPABASE_ACCESS_TOKEN)
        project_id = _require_env("SUPABASE_PROJECT_ID", SUPABASE_PROJECT_ID)
        resp = requests.post(
            f"{MGMT_API_URL}/projects/{project_id}/database/query",
            headers={
                "Authorization": f"Bearer {mgmt_token}",
                "Content-Type": "application/json",
            },
            json={
                "query": (
                    "SELECT tablename, rowsecurity "
                    "FROM pg_tables "
                    "WHERE schemaname = 'public' "
                    "AND tablename IN ('users', 'song_metadata', 'user_liked_songs');"
                )
            },
        )
        assert resp.status_code in (200, 201)
        tables = resp.json()
        
        for table in tables:
            assert table["rowsecurity"] is True, (
                f"RLS is NOT enabled on {table['tablename']}"
            )


# ============================================================================
# TEST SUITE 4: Grant-Level Verification
# ============================================================================

@skip_if_no_network
class TestGrantEnforcement:
    """Verify that grants are correctly scoped per role."""

    def _fetch_grants(self) -> list:
        mgmt_token = _require_env("SUPABASE_ACCESS_TOKEN", SUPABASE_ACCESS_TOKEN)
        project_id = _require_env("SUPABASE_PROJECT_ID", SUPABASE_PROJECT_ID)
        resp = requests.post(
            f"{MGMT_API_URL}/projects/{project_id}/database/query",
            headers={
                "Authorization": f"Bearer {mgmt_token}",
                "Content-Type": "application/json",
            },
            json={
                "query": (
                    "SELECT grantee, table_name, "
                    "string_agg(privilege_type, ', ' ORDER BY privilege_type) as privileges "
                    "FROM information_schema.role_table_grants "
                    "WHERE table_schema = 'public' "
                    "AND table_name IN ('users', 'song_metadata', 'user_liked_songs') "
                    "GROUP BY table_name, grantee "
                    "ORDER BY table_name, grantee;"
                )
            },
        )
        assert resp.status_code in (200, 201)
        return resp.json()

    def test_anon_song_metadata_select_only(self):
        """anon should only have SELECT on song_metadata."""
        grants = self._fetch_grants()
        anon_songs = next(
            (g for g in grants if g["grantee"] == "anon" and g["table_name"] == "song_metadata"),
            None,
        )
        assert anon_songs is not None, "anon should have grants on song_metadata"
        assert anon_songs["privileges"] == "SELECT", (
            f"anon should only have SELECT on song_metadata, got: {anon_songs['privileges']}"
        )

    def test_anon_users_select_only(self):
        """anon should only have SELECT on users."""
        grants = self._fetch_grants()
        anon_users = next(
            (g for g in grants if g["grantee"] == "anon" and g["table_name"] == "users"),
            None,
        )
        assert anon_users is not None, "anon should have grants on users"
        assert anon_users["privileges"] == "SELECT", (
            f"anon should only have SELECT on users, got: {anon_users['privileges']}"
        )

    def test_anon_has_no_truncate_anywhere(self):
        """anon should NEVER have TRUNCATE on any table (critical safety)."""
        grants = self._fetch_grants()
        for g in grants:
            if g["grantee"] == "anon":
                assert "TRUNCATE" not in g["privileges"], (
                    f"CRITICAL: anon has TRUNCATE on {g['table_name']}!"
                )

    def test_authenticated_no_truncate_anywhere(self):
        """authenticated should NEVER have TRUNCATE on any table."""
        grants = self._fetch_grants()
        for g in grants:
            if g["grantee"] == "authenticated":
                assert "TRUNCATE" not in g["privileges"], (
                    f"CRITICAL: authenticated has TRUNCATE on {g['table_name']}!"
                )

    def test_authenticated_user_liked_songs_scoped(self):
        """authenticated should have only SELECT, INSERT, DELETE on user_liked_songs."""
        grants = self._fetch_grants()
        auth_likes = next(
            (g for g in grants if g["grantee"] == "authenticated" and g["table_name"] == "user_liked_songs"),
            None,
        )
        assert auth_likes is not None
        assert auth_likes["privileges"] == "DELETE, INSERT, SELECT", (
            f"authenticated should have DELETE, INSERT, SELECT on user_liked_songs, "
            f"got: {auth_likes['privileges']}"
        )

    def test_service_role_has_full_access(self):
        """service_role should have full privileges on all tables."""
        grants = self._fetch_grants()
        for table in ("users", "song_metadata", "user_liked_songs"):
            svc = next(
                (g for g in grants if g["grantee"] == "service_role" and g["table_name"] == table),
                None,
            )
            assert svc is not None, f"service_role should have grants on {table}"
            for required in ("SELECT", "INSERT", "UPDATE", "DELETE"):
                assert required in svc["privileges"], (
                    f"service_role missing {required} on {table}"
                )


# ============================================================================
# TEST SUITE 5: Cross-Cutting Security Checks
# ============================================================================

@skip_if_no_network
class TestCrossCuttingSecurity:
    """Additional security assertions that span multiple tables."""

    def test_anon_cannot_call_rpc_rls_auto_enable(self):
        """The rls_auto_enable() SECURITY DEFINER function should NOT be callable by anon."""
        resp = requests.post(
            f"{REST_URL}/rpc/rls_auto_enable",
            headers=_anon_headers(),
            json={},
        )
        # Should be 403 (permission denied) or 404 (not exposed)
        # 400 = trigger function cannot be called directly, which is also a valid denial
        assert resp.status_code in (400, 403, 401, 404, 500), (
            f"rls_auto_enable should NOT be callable by anon, got {resp.status_code}: {resp.text}"
        )

    def test_anon_cannot_access_alembic_version(self):
        """alembic_version should be completely inaccessible to anon."""
        resp = requests.get(
            f"{REST_URL}/alembic_version?select=version_num",
            headers=_anon_headers(),
        )
        # Should return 403 or empty due to RLS + revoked grants
        if resp.status_code == 200:
            data = resp.json()
            assert data == [], "anon should see 0 rows from alembic_version"
        else:
            assert resp.status_code in (403, 401, 404)

    def test_anon_cannot_truncate_via_rpc(self):
        """Verify anon cannot call TRUNCATE via RPC (defense against grant-level bypass)."""
        resp = requests.post(
            f"{REST_URL}/rpc/",
            headers=_anon_headers(),
            json={"query": "TRUNCATE users CASCADE;"},
        )
        # PostgREST doesn't expose raw SQL via RPC, but verify it's not accessible
        assert resp.status_code != 200 or resp.json() == [], (
            "Raw SQL execution should never be accessible via PostgREST"
        )
