"""Rewrite the variant/condition fields inside stored IVA products after the rename.

An .npz is a zip of .npy members. Only two tiny members need new contents, so the large
compressed arrays are copied through byte for byte rather than decompressed and
recompressed -- a 1.1 GB DEFLATE member would otherwise dominate the whole job.

Writes to a sibling temp file and renames, so an interrupted run cannot leave a
half-written product behind.
"""

import io
import os
import shutil
import struct
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

REPLACEMENTS = {
    "variant": {
        "iva_channel_joined": "iva_channel_joined_separate",
        "ica_channel_joined": "ica_channel_joined_separate",
    },
    "condition": {"Joined": "JoinedSeparate"},
}


def _raw_member(handle, info: zipfile.ZipInfo) -> bytes:
    """The member's stored bytes, exactly as they sit in the archive."""
    handle.seek(info.header_offset)
    signature, _, _, _, _, _, _, _, _, name_len, extra_len = struct.unpack(
        "<IHHHHHIIIHH", handle.read(30)
    )
    if signature != 0x04034B50:
        raise ValueError(f"{info.filename}: not a local file header")
    handle.seek(info.header_offset + 30 + name_len + extra_len)
    return handle.read(info.compress_size)


def migrate(path: Path) -> bool:
    """Rewrite *path* in place if it carries a pre-rename value. Returns True if changed."""
    with np.load(path) as cached:
        current = {k: str(cached[k]) for k in ("variant", "condition") if k in cached}
    new_values = {
        key: REPLACEMENTS[key][value]
        for key, value in current.items()
        if value in REPLACEMENTS.get(key, {})
    }
    if not new_values:
        return False

    handle, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npz.tmp")
    os.close(handle)
    try:
        with open(path, "rb") as source, zipfile.ZipFile(path) as src_zip:
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
                for info in src_zip.infolist():
                    field = info.filename.removesuffix(".npy")
                    if field in new_values:
                        buffer = io.BytesIO()
                        np.save(buffer, np.array(new_values[field]))
                        out.writestr(field + ".npy", buffer.getvalue())
                        continue
                    # copy the stored bytes straight through
                    kept = zipfile.ZipInfo(info.filename, info.date_time)
                    kept.compress_type = info.compress_type
                    kept.external_attr = info.external_attr
                    kept.file_size = info.file_size
                    kept.compress_size = info.compress_size
                    kept.CRC = info.CRC
                    raw = _raw_member(source, info)
                    with out._lock:  # noqa: SLF001 - writing a pre-compressed member
                        kept.header_offset = out.fp.tell()
                        out.fp.write(kept.FileHeader(False))
                        out.fp.write(raw)
                        out.start_dir = out.fp.tell()
                        out._end_offset = out.start_dir
                        out.filelist.append(kept)
                        out.NameToInfo[kept.filename] = kept
        shutil.copystat(path, tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    print(f"  {path.name}: {current} -> {new_values}")
    return True


if __name__ == "__main__":
    root = Path(sys.argv[1])
    changed = sum(migrate(p) for p in sorted(root.rglob("*.npz")))
    print(f"\n{changed} file(s) migrated")
