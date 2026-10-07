import os
from pathlib import Path

from dotenv import load_dotenv


def load_local_env() -> None:
    """Load the repository-root .env without overriding shell variables."""
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / ".env").is_file():
            load_dotenv(p / ".env", override=False)
            return
        p = p.parent


def get_provider_credentials() -> dict[str, str]:
    load_local_env()
    provider = os.getenv("MARKET_DATA_PROVIDER", "upstox").strip().lower()
    if provider == "angelone":
        names = (
            "ANGELONE_API_KEY",
            "ANGELONE_CLIENT_CODE",
            "ANGELONE_PASSWORD",
            "ANGELONE_TOTP_SECRET",
        )
    else:
        raise ValueError("MARKET_DATA_PROVIDER must be 'angelone'")
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            f"Missing credentials for {provider}: {', '.join(missing)}. "
            "Set them in the repo-root .env or your shell environment."
        )
    return {name: os.environ[name] for name in names}
