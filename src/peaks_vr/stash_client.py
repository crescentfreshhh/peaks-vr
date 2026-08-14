"""Minimal Stash GraphQL client.

Step 1 scope:
  - connection / version check
  - count + iterate scenes (paginated) with their files and existing markers
  - helper to find-or-create a tag and create a scene marker (used in step 3)

Auth: Stash uses an `ApiKey` request header. If your server has no
authentication configured, leave the key empty.
"""

from __future__ import annotations

import time
from typing import Any, Iterator

import requests

from .models import Scene, Tag


class StashError(RuntimeError):
    """Raised when Stash returns a GraphQL error or an unexpected response."""


# --- GraphQL documents -------------------------------------------------------

_VERSION_QUERY = """
query Version {
  version { version build_time hash }
}
"""

# A page of scenes with the fields we care about for sampling + marker reads.
_FIND_SCENES_QUERY = """
query FindScenes($filter: FindFilterType) {
  findScenes(filter: $filter) {
    count
    scenes {
      id
      title
      files {
        path
        duration
        width
        height
        frame_rate
        video_codec
        size
        fingerprints { type value }
      }
      scene_markers {
        id
        seconds
        end_seconds
        title
        primary_tag { id name }
      }
    }
  }
}
"""

_FIND_SCENE_MARKERS_QUERY = """
query FindSceneMarkers($filter: FindFilterType, $marker_filter: SceneMarkerFilterType) {
  findSceneMarkers(filter: $filter, scene_marker_filter: $marker_filter) {
    count
    scene_markers {
      id
      seconds
      end_seconds
      title
      scene { id }
      primary_tag { id name }
    }
  }
}
"""

_SCENE_DETAILS_QUERY = """
query SceneDetails($ids: [ID!]) {
  findScenes(ids: $ids, filter: {per_page: -1}) {
    scenes {
      id
      title
      date
      details
      rating100
      o_counter
      organized
      studio { name }
      performers { id name gender }
      tags { name }
      files { path duration width height }
      paths { screenshot preview }
    }
  }
}
"""

_SCENES_BY_PERFORMER_QUERY = """
query ScenesByPerformer($pid: ID!, $per_page: Int!) {
  findScenes(
    filter: {per_page: $per_page}
    scene_filter: {performers: {value: [$pid], modifier: INCLUDES}}
  ) {
    scenes { id }
  }
}
"""

_FIND_PERFORMERS_QUERY = """
query FindPerformers($filter: FindFilterType, $performer_filter: PerformerFilterType) {
  findPerformers(filter: $filter, performer_filter: $performer_filter) {
    performers { id name image_path scene_count }
  }
}
"""

_FIND_TAGS_QUERY = """
query FindTags($filter: FindFilterType, $tag_filter: TagFilterType) {
  findTags(filter: $filter, tag_filter: $tag_filter) {
    tags { id name }
  }
}
"""

_TAG_CREATE = """
mutation TagCreate($input: TagCreateInput!) {
  tagCreate(input: $input) { id name }
}
"""

_SCENE_MARKER_CREATE = """
mutation SceneMarkerCreate($input: SceneMarkerCreateInput!) {
  sceneMarkerCreate(input: $input) { id seconds end_seconds title }
}
"""

_SCENE_UPDATE = """
mutation SceneUpdate($input: SceneUpdateInput!) {
  sceneUpdate(input: $input) {
    id rating100 organized o_counter title date details
  }
}
"""

_SCENE_ADD_O = "mutation AddO($id: ID!) { sceneAddO(id: $id) { count } }"
_SCENE_DELETE_O = "mutation DelO($id: ID!) { sceneDeleteO(id: $id) { count } }"
_SCENE_RESET_O = "mutation ResetO($id: ID!) { sceneResetO(id: $id) }"

_SCENE_MARKERS_DESTROY = """
mutation SceneMarkersDestroy($ids: [ID!]!) {
  sceneMarkersDestroy(ids: $ids)
}
"""

_BULK_SCENE_UPDATE = """
mutation BulkSceneUpdate($input: BulkSceneUpdateInput!) {
  bulkSceneUpdate(input: $input) { id }
}
"""


class StashClient:
    # transient network errors are retried with these sleeps between attempts;
    # a multi-hour embed run pages the scene list lazily and shouldn't die at
    # hour five because one request hit a blip. GraphQL/HTTP errors (bad auth,
    # bad query) are NOT retried — those won't fix themselves.
    RETRY_SLEEPS: tuple[float, ...] = (1.0, 4.0, 10.0)

    def __init__(self, url: str, api_key: str = "", timeout: int = 30):
        self.base_url = url.rstrip("/")
        self.graphql_url = f"{self.base_url}/graphql"
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers["ApiKey"] = api_key

    # --- core ----------------------------------------------------------------

    def execute(self, query: str, variables: dict | None = None) -> dict[str, Any]:
        for attempt, sleep in enumerate((*self.RETRY_SLEEPS, None)):
            try:
                resp = self.session.post(
                    self.graphql_url,
                    json={"query": query, "variables": variables or {}},
                    timeout=self.timeout,
                )
                break
            except requests.RequestException as exc:
                if sleep is None:
                    raise StashError(
                        f"Could not reach Stash at {self.graphql_url} "
                        f"(after {attempt + 1} attempts): {exc}"
                    ) from exc
                time.sleep(sleep)

        if resp.status_code != 200:
            raise StashError(
                f"Stash returned HTTP {resp.status_code}: {resp.text[:500]}"
            )
        payload = resp.json()
        if payload.get("errors"):
            raise StashError(f"GraphQL error: {payload['errors']}")
        return payload["data"]

    # --- reads ---------------------------------------------------------------

    def version(self) -> dict:
        return self.execute(_VERSION_QUERY)["version"]

    def scene_count(self) -> int:
        data = self.execute(
            _FIND_SCENES_QUERY,
            {"filter": {"per_page": 0}},
        )
        return data["findScenes"]["count"]

    @staticmethod
    def _under_path(scene: Scene, prefix: str) -> bool:
        """True if the scene's file is inside `prefix` (folder, incl. subdirs).
        Client-side so it works regardless of Stash's filter schema."""
        if not prefix:
            return True
        p = scene.path
        if not p:
            return False
        base = prefix.rstrip("/")
        return p == base or p.startswith(base + "/")

    def iter_scenes(
        self, page_size: int = 100, path_prefix: str = ""
    ) -> Iterator[Scene]:
        """Yield every scene, transparently paginating. When `path_prefix` is
        set, only scenes whose file lives under that folder are yielded."""
        page = 1
        seen = 0
        while True:
            data = self.execute(
                _FIND_SCENES_QUERY,
                {
                    "filter": {
                        "per_page": page_size,
                        "page": page,
                        "sort": "id",
                        "direction": "ASC",
                    }
                },
            )
            result = data["findScenes"]
            scenes = result["scenes"]
            for s in scenes:
                scene = Scene.from_dict(s)
                if self._under_path(scene, path_prefix):
                    yield scene
            seen += len(scenes)
            if not scenes or seen >= result["count"]:
                break
            page += 1

    # --- scene writes (two-way sync with Stash) -----------------------------

    _EDITABLE = ("rating100", "organized", "title", "date", "details")

    def update_scene(self, scene_id: str, **fields) -> dict:
        """Update editable scene fields in Stash (only the ones passed).
        Returns the updated scene dict."""
        inp: dict = {"id": str(scene_id)}
        for k in self._EDITABLE:
            if k in fields and fields[k] is not None:
                inp[k] = fields[k]
        data = self.execute(_SCENE_UPDATE, {"input": inp})
        return data["sceneUpdate"]

    def scene_add_o(self, scene_id: str) -> int:
        """Record one O; returns the new count."""
        return self.execute(_SCENE_ADD_O, {"id": str(scene_id)})["sceneAddO"]["count"]

    def scene_delete_o(self, scene_id: str) -> int:
        """Remove the most recent O; returns the new count."""
        return self.execute(_SCENE_DELETE_O, {"id": str(scene_id)})["sceneDeleteO"]["count"]

    def scene_reset_o(self, scene_id: str) -> int:
        return int(self.execute(_SCENE_RESET_O, {"id": str(scene_id)})["sceneResetO"])

    def scene_details(self, ids: list[str]) -> dict[str, dict]:
        """Fetch display metadata for scene ids → {id: {title, performers,
        studio, date, tags, rating, cover, duration}}. Empty ids → {}."""
        ids = [str(i) for i in ids if i]
        if not ids:
            return {}
        data = self.execute(_SCENE_DETAILS_QUERY, {"ids": ids})
        out: dict[str, dict] = {}
        for s in data["findScenes"]["scenes"]:
            files = s.get("files") or []
            f0 = files[0] if files else {}
            paths = s.get("paths") or {}
            out[str(s["id"])] = {
                "title": s.get("title") or "",
                "date": s.get("date") or "",
                "details": s.get("details") or "",
                "rating100": s.get("rating100"),
                "o_counter": s.get("o_counter") or 0,
                "organized": bool(s.get("organized")),
                "studio": (s.get("studio") or {}).get("name") or "",
                "performers": [p.get("name", "") for p in (s.get("performers") or [])],
                "performers_detail": [
                    {"id": p.get("id"), "name": p.get("name", "")}
                    for p in (s.get("performers") or [])
                ],
                "tags": [t.get("name", "") for t in (s.get("tags") or [])],
                "duration": f0.get("duration"),
                "width": f0.get("width"),
                "height": f0.get("height"),
                "path": f0.get("path"),
                "cover": paths.get("screenshot"),
            }
        return out

    def scenes_for_performer(self, performer_id: str, limit: int = 500) -> list[str]:
        """Scene ids featuring a performer — powers 'more from this actress'."""
        if not performer_id:
            return []
        data = self.execute(
            _SCENES_BY_PERFORMER_QUERY, {"pid": str(performer_id), "per_page": limit}
        )
        return [str(s["id"]) for s in data["findScenes"]["scenes"]]

    def find_performers(self, name: str | None = None, limit: int = 500) -> list[dict]:
        """Performers as [{id, name, image, scene_count}]. With `name`, fuzzy-match
        it (regex); otherwise return the library's performers (for the leaderboard)."""
        pf = None
        if name:
            pf = {"name": {"value": name, "modifier": "MATCHES_REGEX"}}
        data = self.execute(
            _FIND_PERFORMERS_QUERY,
            {"filter": {"per_page": limit}, "performer_filter": pf},
        )
        out = []
        for p in data["findPerformers"]["performers"]:
            out.append({
                "id": str(p["id"]), "name": p.get("name", ""),
                "image": p.get("image_path"), "scene_count": p.get("scene_count", 0),
            })
        return out

    def performer_image(self, performer_id: str) -> tuple[bytes, str] | None:
        """Fetch a performer's Stash image bytes (+content-type) through the authed
        session, so the web app can proxy it without leaking the api key."""
        try:
            r = self.session.get(
                f"{self.base_url}/performer/{performer_id}/image", timeout=self.timeout
            )
            r.raise_for_status()
        except Exception:  # noqa: BLE001 — missing image is not fatal
            return None
        return r.content, r.headers.get("content-type", "image/jpeg")

    def iter_markers_by_tag(
        self, tag_name: str, page_size: int = 200
    ) -> Iterator[dict]:
        """Yield scene markers whose tags include `tag_name`.

        Each yielded dict: {marker_id, scene_id, seconds, end_seconds, title}.
        Returns nothing if the tag doesn't exist yet.
        """
        tag = self.find_tag_by_name(tag_name)
        if tag is None:
            return
        marker_filter = {
            "tags": {"value": [tag.id], "modifier": "INCLUDES", "depth": 0}
        }
        page = 1
        seen = 0
        while True:
            data = self.execute(
                _FIND_SCENE_MARKERS_QUERY,
                {
                    "filter": {
                        "per_page": page_size,
                        "page": page,
                        "sort": "scene_id",
                        "direction": "ASC",
                    },
                    "marker_filter": marker_filter,
                },
            )
            result = data["findSceneMarkers"]
            markers = result["scene_markers"]
            for m in markers:
                scene = m.get("scene") or {}
                pt = m.get("primary_tag") or {}
                yield {
                    "marker_id": str(m["id"]),
                    "scene_id": str(scene.get("id")) if scene.get("id") else None,
                    "seconds": float(m.get("seconds") or 0.0),
                    "end_seconds": (
                        float(m["end_seconds"]) if m.get("end_seconds") else None
                    ),
                    "title": m.get("title", ""),
                    "primary_tag": pt.get("name", ""),
                }
            seen += len(markers)
            if not markers or seen >= result["count"]:
                break
            page += 1

    # --- writes (used in step 3) --------------------------------------------

    def find_tag_by_name(self, name: str) -> Tag | None:
        data = self.execute(
            _FIND_TAGS_QUERY,
            {
                "filter": {"per_page": 1},
                "tag_filter": {"name": {"value": name, "modifier": "EQUALS"}},
            },
        )
        tags = data["findTags"]["tags"]
        return Tag.from_dict(tags[0]) if tags else None

    def find_or_create_tag(self, name: str) -> Tag:
        existing = self.find_tag_by_name(name)
        if existing:
            return existing
        data = self.execute(_TAG_CREATE, {"input": {"name": name}})
        return Tag.from_dict(data["tagCreate"])

    def create_scene_marker(
        self,
        scene_id: str,
        seconds: float,
        primary_tag_id: str,
        title: str = "",
        end_seconds: float | None = None,
    ) -> dict:
        """Create a marker on a scene. `end_seconds` makes it a ranged marker."""
        input_obj: dict[str, Any] = {
            "scene_id": scene_id,
            "seconds": seconds,
            "primary_tag_id": primary_tag_id,
            "title": title,
        }
        if end_seconds is not None:
            input_obj["end_seconds"] = end_seconds
        data = self.execute(_SCENE_MARKER_CREATE, {"input": input_obj})
        return data["sceneMarkerCreate"]

    def add_scene_tags(self, scene_ids: list[str], tag_ids: list[str]) -> int:
        """Append tags to scenes without clobbering their existing tags
        (bulkSceneUpdate ADD mode). Returns how many scenes were submitted."""
        scene_ids = [str(s) for s in scene_ids if s]
        tag_ids = [str(t) for t in tag_ids if t]
        if not scene_ids or not tag_ids:
            return 0
        self.execute(
            _BULK_SCENE_UPDATE,
            {"input": {"ids": scene_ids, "tag_ids": {"ids": tag_ids, "mode": "ADD"}}},
        )
        return len(scene_ids)

    def destroy_scene_markers(self, marker_ids: list[str], chunk: int = 100) -> int:
        """Delete markers by id (chunked). Returns how many ids were submitted."""
        for i in range(0, len(marker_ids), chunk):
            self.execute(
                _SCENE_MARKERS_DESTROY, {"ids": marker_ids[i : i + chunk]}
            )
        return len(marker_ids)

    # --- playback helpers (used by the megaboard) ---------------------------

    def stream_url(self, scene_id: str, start: float | None = None) -> str:
        """Direct-stream URL for a scene, optionally seeked to `start` seconds.

        The megaboard points <video> tiles at these. `apikey` is appended as a
        query param because <video> tags can't send the ApiKey header.
        """
        url = f"{self.base_url}/scene/{scene_id}/stream"
        params = []
        if start is not None:
            params.append(f"start={start:g}")
        api_key = self.session.headers.get("ApiKey")
        if api_key:
            params.append(f"apikey={api_key}")
        if params:
            url += "?" + "&".join(params)
        return url

    @classmethod
    def from_config(cls, cfg) -> "StashClient":
        return cls(url=cfg.stash.url, api_key=cfg.stash.api_key, timeout=cfg.stash.timeout)
