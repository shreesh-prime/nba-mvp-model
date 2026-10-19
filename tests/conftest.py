import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Fail any outbound connection so tests stay offline; records attempted addresses."""
    attempts = []

    def guarded_connect(self, address, *args, **kwargs):
        attempts.append(address)
        raise RuntimeError(f"Network access blocked in tests: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect)
    return attempts
