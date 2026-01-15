"""
Constants for the Nzambe application.
Contains truly immutable values that never change across environments.
"""

# Application metadata
APP_TITLE = "Nzambe RAG API"
APP_DESCRIPTION = "Question answering system for holy books using RAG"

# Bible book name mappings (for parsing Gutenberg text)
ALTERNATIVE_BOOK_NAMES = {
    "The First Book of Samuel": "The First Book of the Kings",
    "The Second Book of Samuel": "The Second Book of the Kings",
    "The First Book of the Kings": "The Third Book of the Kings",
    "The Second Book of the Kings": "The Fourth Book of the Kings",
    "Ecclesiastes": "The Preacher",
}

# Testament boundaries for parsing
OLD_TESTAMENT_FIRST_BOOK = "The First Book of Moses: Called Genesis"
OLD_TESTAMENT_LAST_BOOK = "Malachi"
NEW_TESTAMENT_FIRST_BOOK = "The Gospel According to Saint Matthew"
NEW_TESTAMENT_LAST_BOOK = "The Revelation of Saint John the Divine"
GUTENBERG_END_MARKER = (
    "*** END OF THE PROJECT GUTENBERG EBOOK THE KING JAMES VERSION OF THE BIBLE ***"
)

# Regex patterns
VERSE_PATTERN = r"(\w+:\w+)"

# FASTAPI server settings
NZAMBE_SERVER_HOST = "localhost"
NZAMBE_SERVER_PORT = 8000
NZAMBE_SERVER_DEFAULT_BASE_URL = f"http://{NZAMBE_SERVER_HOST}:{NZAMBE_SERVER_PORT}"
