import os

# Set the environment variable to 'test' before any other imports happen.
# This ensures that when your config.py is imported, it sees NZAMBE_ENV=test
os.environ["NZAMBE_ENV"] = "test"
