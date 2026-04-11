from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

try:
    import fcntl
except ImportError:  # non-Unix
    fcntl = None  # type: ignore


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _merge_capability_disk(base: Dict, overlay: Dict) -> Dict:
    """Merge overlay into a copy of base (union blob keys; overlay wins on scalar conflicts)."""
    out: Dict = json.loads(json.dumps(base)) if base else {}
    out.setdefault("version", 1)
    out.setdefault("sources", {})
    out.setdefault("models", {})
    ov = overlay or {}
    for sid, srec in ov.get("sources", {}).items():
        t = out["sources"].setdefault(sid, {})
        for k, v in srec.items():
            if k == "observed_blob_keys":
                known = set(t.get("observed_blob_keys", []))
                known.update(v if isinstance(v, list) else [])
                t["observed_blob_keys"] = sorted(known)
            elif k == "observed_optional_channels":
                opt = t.setdefault("observed_optional_channels", {})
                for alias, val in (v or {}).items():
                    if val:
                        opt[alias] = True
                    else:
                        opt.setdefault(alias, False)
            else:
                t[k] = v
    for sid, mblock in ov.get("models", {}).items():
        tgt = out["models"].setdefault(sid, {})
        for mname, mrec in (mblock or {}).items():
            cur = tgt.setdefault(mname, {})
            cur.update(mrec)
    return out


class CapabilityMemoryStore:
    """Persist per-source and per-model channel capability observations."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.path.with_name(self.path.name + ".lock")
        self._db = self._load()

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        if fcntl is None:
            yield
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._lock_path, "a+", encoding="utf-8") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    def _load(self) -> Dict:
        if not self.path.is_file():
            return {"version": 1, "sources": {}, "models": {}}
        try:
            with self._file_lock():
                if not self.path.is_file():
                    return {"version": 1, "sources": {}, "models": {}}
                return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 1, "sources": {}, "models": {}}

    def _save(self) -> None:
        payload = json.dumps(self._db, ensure_ascii=False, indent=2, sort_keys=True)
        with self._file_lock():
            on_disk: Dict = {"version": 1, "sources": {}, "models": {}}
            if self.path.is_file():
                try:
                    on_disk = json.loads(self.path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            merged = _merge_capability_disk(on_disk, self._db)
            self._db = merged
            tmp = self.path.with_name(f".{os.getpid()}.{self.path.name}.tmp")
            tmp.write_text(
                json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(str(tmp), str(self.path))

    @staticmethod
    def make_source_id(
        source_name: str,
        root: Path,
        fmt: Optional[str],
        adapter_name: str,
    ) -> str:
        return f"{source_name}|{fmt or 'auto'}|{adapter_name}|{Path(root)}"

    def ensure_source(
        self,
        source_id: str,
        source_name: str,
        root: Path,
        fmt: Optional[str],
        adapter_name: str,
    ) -> None:
        src = self._db["sources"].setdefault(source_id, {})
        src.update(
            {
                "source_name": source_name,
                "root": str(root),
                "format": fmt or "auto",
                "adapter": adapter_name,
                "updated_at": _utc_now(),
            }
        )
        src.setdefault("observed_blob_keys", [])
        src.setdefault("observed_optional_channels", {})
        self._save()

    def observe_blob_keys(
        self,
        source_id: str,
        blob_keys: List[str],
        optional_aliases: Dict[str, str],
    ) -> None:
        src = self._db["sources"].setdefault(source_id, {})
        known = set(src.setdefault("observed_blob_keys", []))
        for k in blob_keys:
            known.add(k)
        src["observed_blob_keys"] = sorted(known)
        obs_opt = src.setdefault("observed_optional_channels", {})
        for alias, bkey in optional_aliases.items():
            if bkey in blob_keys:
                obs_opt[alias] = True
            else:
                obs_opt.setdefault(alias, False)
        src["updated_at"] = _utc_now()
        self._save()

    def is_optional_available(self, source_id: str, alias: str) -> bool:
        src = self._db["sources"].get(source_id, {})
        return bool(src.get("observed_optional_channels", {}).get(alias, False))

    def update_model_decision(
        self,
        source_id: str,
        model_name: str,
        enabled_channels: List[str],
        missing_channels: List[str],
        fallback_used: List[Dict],
    ) -> None:
        models = self._db["models"].setdefault(source_id, {})
        rec = models.setdefault(model_name, {})
        rec.update(
            {
                "enabled_channels": sorted(set(enabled_channels)),
                "missing_channels": sorted(set(missing_channels)),
                "fallback_used": fallback_used,
                "updated_at": _utc_now(),
            }
        )
        self._save()

    def get_source_record(self, source_id: str) -> Dict:
        return dict(self._db["sources"].get(source_id, {}))

