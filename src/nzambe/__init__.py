"""Nzambe - Question answering system for holy books using RAG."""

try:
    from nzambe._version import __version__
except ImportError:
    # Fallback when _version.py doesn't exist (e.g., not installed or git repo unavailable)
    __version__ = "0.0.0.dev0+unknown"
