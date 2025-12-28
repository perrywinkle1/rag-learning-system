"""Pytest fixtures for Closed-Loop RAG Learning System tests."""

import asyncio
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
