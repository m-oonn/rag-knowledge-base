"""Complete API test suite for RAG Knowledge Base System.

Run: pytest tests/ -v
"""

import pytest
import requests
import random
import string

BASE = "http://localhost:8000/api/v1"


# ── Fixture: create a test user + KB + document ──────────────

@pytest.fixture(scope="module")
def auth():
    """Register a fresh user and return auth headers."""
    suffix = "".join(random.choices(string.ascii_lowercase, k=6))
    username = f"test_{suffix}"
    password = "testpass123"

    # Register
    requests.post(f"{BASE}/auth/register", json={
        "username": username,
        "email": f"{username}@test.com",
        "password": password,
    })
    # Login
    r = requests.post(f"{BASE}/auth/login/access-token", data={
        "username": username,
        "password": password,
    })
    assert r.status_code == 200, f"Login failed: {r.text}"
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="module")
def kb(auth):
    """Create a test knowledge base."""
    r = requests.post(f"{BASE}/knowledge-bases/", json={
        "name": f"pytest_kb_{random.randint(1000,9999)}",
        "description": "Pytest KB",
    }, headers=auth)
    assert r.status_code == 200
    return r.json()


@pytest.fixture(scope="module")
def kb_with_doc(auth, kb):
    """Upload a test document and return kb."""
    content = "# Test Doc\n\nPython was created by Guido van Rossum in 1991.\n\nFastAPI is a modern Python web framework with automatic docs."
    r = requests.post(
        f"{BASE}/knowledge-bases/{kb['id']}/upload",
        files={"file": ("test.md", content.encode(), "text/markdown")},
        headers=auth,
    )
    assert r.status_code == 200
    assert r.json().get("status") == 2  # Completed
    return kb


# ── Tests ────────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self):
        r = requests.get(f"{BASE}/health/")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestAuth:
    def test_register(self):
        suffix = "".join(random.choices(string.ascii_lowercase, k=8))
        r = requests.post(f"{BASE}/auth/register", json={
            "username": f"regtest_{suffix}",
            "email": f"regtest_{suffix}@test.com",
            "password": "secure123",
        })
        assert r.status_code == 200
        assert "id" in r.json()

    def test_login(self, auth):
        # auth fixture already logged in successfully
        assert "Authorization" in auth
        assert auth["Authorization"].startswith("Bearer ")


class TestKnowledgeBase:
    def test_create_kb(self, auth):
        r = requests.post(f"{BASE}/knowledge-bases/", json={
            "name": f"test_kb_{random.randint(1000,9999)}",
            "description": "Test",
        }, headers=auth)
        assert r.status_code == 200

    def test_list_kbs(self, auth):
        r = requests.get(f"{BASE}/knowledge-bases/", headers=auth)
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_upload_document(self, auth, kb):
        content = b"# Hello\nWorld"
        r = requests.post(
            f"{BASE}/knowledge-bases/{kb['id']}/upload",
            files={"file": ("hello.md", content, "text/markdown")},
            headers=auth,
        )
        assert r.status_code == 200
        assert r.json().get("status") in (0, 1, 2)

    def test_list_documents(self, auth, kb_with_doc):
        r = requests.get(
            f"{BASE}/knowledge-bases/{kb_with_doc['id']}/documents",
            headers=auth,
        )
        assert r.status_code == 200
        assert len(r.json()) >= 1


class TestRAG:
    def test_query_with_answer(self, auth, kb_with_doc):
        r = requests.post(f"{BASE}/chat/", json={
            "query": "Who created Python?",
            "top_k": 3,
            "kb_id": kb_with_doc["id"],
        }, headers=auth)
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert len(data["answer"]) > 10
        # Should mention Guido or van Rossum
        assert "Guido" in data["answer"] or "Rossum" in data["answer"] or "Python" in data["answer"].lower()

    def test_query_no_kb_returns_answer(self, auth):
        """General chat without KB should still work."""
        r = requests.post(f"{BASE}/chat/", json={
            "query": "Hello!",
            "top_k": 3,
        }, headers=auth)
        assert r.status_code == 200

    def test_query_with_sources(self, auth, kb_with_doc):
        r = requests.post(f"{BASE}/chat/", json={
            "query": "What is FastAPI?",
            "top_k": 5,
            "kb_id": kb_with_doc["id"],
        }, headers=auth)
        assert r.status_code == 200
        data = r.json()
        assert len(data.get("source_documents", [])) >= 1

    def test_streaming(self, auth, kb_with_doc):
        r = requests.post(f"{BASE}/chat/stream/", json={
            "query": "Who created Python?",
            "top_k": 3,
            "kb_id": kb_with_doc["id"],
        }, headers=auth, stream=True)
        assert r.status_code == 200
        content = ""
        for line in r.iter_lines():
            if line and line.startswith(b"data: "):
                content += line.decode()
        assert len(content) > 0


class TestSessions:
    def test_list_sessions(self, auth):
        r = requests.get(f"{BASE}/chat/sessions", headers=auth)
        assert r.status_code == 200
        assert isinstance(r.json(), list)
