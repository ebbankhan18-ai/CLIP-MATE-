"""Publishing to social platforms. Runs entirely on your machine: tokens are stored in ./data.

Implemented: YouTube (Shorts are just vertical videos under 3 minutes).
Adding another platform = one function that takes a file path + metadata and returns a URL.
"""
import os
from pathlib import Path

# Allow the OAuth redirect to http://localhost (this app never leaves your machine).
os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
YT_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
_pending = {}  # oauth state -> Flow (keeps the PKCE verifier between login and callback)


def _secrets_path():
    p = Path(os.getenv("YOUTUBE_CLIENT_SECRETS", "client_secret.json"))
    return p if p.is_absolute() else ROOT / p


def _token_path():
    return DATA / "youtube_token.json"


def youtube_status():
    return {"configured": _secrets_path().exists(), "connected": _token_path().exists()}


def youtube_auth_url(redirect_uri):
    from google_auth_oauthlib.flow import Flow
    if not _secrets_path().exists():
        raise RuntimeError("client_secret.json not found. See README, step 'Connect YouTube'.")
    flow = Flow.from_client_secrets_file(str(_secrets_path()), scopes=YT_SCOPES, redirect_uri=redirect_uri)
    url, state = flow.authorization_url(access_type="offline", prompt="consent")
    _pending[state] = flow
    return url


def youtube_finish_auth(state, full_url):
    flow = _pending.pop(state, None)
    if flow is None:
        raise RuntimeError("Login session expired. Click Connect YouTube again.")
    flow.fetch_token(authorization_response=full_url)
    _token_path().write_text(flow.credentials.to_json(), encoding="utf-8")


def youtube_disconnect():
    if _token_path().exists():
        _token_path().unlink()


def _credentials():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    if not _token_path().exists():
        raise RuntimeError("YouTube is not connected yet.")
    creds = Credentials.from_authorized_user_file(str(_token_path()), YT_SCOPES)
    if not creds.valid:
        creds.refresh(Request())
        _token_path().write_text(creds.to_json(), encoding="utf-8")
    return creds


def youtube_upload(path, title, description="", privacy="private", tags=None):
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    yt = build("youtube", "v3", credentials=_credentials(), cache_discovery=False)
    if "#shorts" not in (title + description).lower():
        description = (description + "\n\n#Shorts").strip()
    body = {
        "snippet": {"title": title[:100], "description": description[:4900], "tags": tags or [], "categoryId": "22"},
        "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(str(path), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    return {"id": resp["id"], "url": f"https://youtube.com/shorts/{resp['id']}"}
