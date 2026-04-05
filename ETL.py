import praw
import os
from dotenv import load_dotenv

api = os.getenv("REDDIT_API")

reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret=None,  # Use None for implicit flow
    user_agent="YourAppName v1.0"
)

# Replace 'access_token_here' with your token
# 'expires_in' is the seconds remaining; 'scope' is usually "*" for all
reddit.auth.implicit(access_token="access_token_here", expires_in=3600, scope="*")

print(reddit.user.me()) # Verify it works