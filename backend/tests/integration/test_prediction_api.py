import pytest
from httpx import ASGITransport, AsyncClient

from main import app

pytestmark = pytest.mark.skip(reason="Scaffold — implement in Phase 8")


@pytest.mark.asyncio
async def test_health_placeholder():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")
    assert response.status_code == 200
