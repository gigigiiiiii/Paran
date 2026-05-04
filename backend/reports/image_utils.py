from __future__ import annotations

from typing import Any
from urllib.parse import quote


def snapshot_public_url(
    supabase_url: str | None,
    bucket: str | None,
    snapshot_path: str | None,
) -> str | None:
    """Build a public Supabase Storage URL for a saved event snapshot."""
    if not supabase_url or not bucket or not snapshot_path:
        return None
    base_url = supabase_url.rstrip("/")
    bucket_part = quote(bucket.strip("/"), safe="")
    path_part = quote(snapshot_path.strip("/"), safe="/")
    return f"{base_url}/storage/v1/object/public/{bucket_part}/{path_part}"


def attach_snapshot_urls(
    events: list[dict[str, Any]],
    supabase_url: str | None,
    bucket: str | None,
) -> list[dict[str, Any]]:
    """Return copied events with `snapshot_url` filled when storage info exists."""
    enriched: list[dict[str, Any]] = []
    for event in events:
        copied = dict(event)
        copied["snapshot_url"] = snapshot_public_url(
            supabase_url,
            bucket,
            str(event.get("snapshot_path") or ""),
        )
        enriched.append(copied)
    return enriched
