from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)

def setup_module(module):
    global client
    client = TestClient(app)
    client.__enter__()

def teardown_module(module):
    client.__exit__(None, None, None)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_query_happy_path():
    response = client.post("/query", json={"question": "What is predictive analytics?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)

def test_query_invalid_input():
    response = client.post("/query", json={})
    assert response.status_code == 422