import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from tqdm import tqdm


def load_gcs_metadata(gcs_metadata_path: str) -> List[dict]:
    """Load metadata.jsonl from a GCS path."""
    import gcsfs
    fs = gcsfs.GCSFileSystem()
    print(f"Loading GCS metadata: {gcs_metadata_path}")
    with fs.open(gcs_metadata_path, "r") as f:
        entries = [json.loads(line) for line in f if line.strip()]
    print(f"Found {len(entries)} entries")
    return entries


def download_audio_parallel(entries: List[dict], cache_dir: str, workers: int = 64) -> dict:
    """Download unique audio files from GCS in parallel.

    Returns:
        audio_map: dict mapping audio URL -> local file path
    """
    import gcsfs
    fs = gcsfs.GCSFileSystem()
    audio_dir = Path(cache_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Deduplicate audio URLs
    audio_urls = set()
    for entry in entries:
        url = entry.get("audio", "")
        if url:
            audio_urls.add(url)

    print(f"Total entries: {len(entries)}, Unique audio files: {len(audio_urls)}")

    # Check what's already downloaded
    def url_to_path(url):
        return audio_dir / hashlib.md5(url.encode()).hexdigest()

    already = sum(1 for url in audio_urls if url_to_path(url).exists())
    to_download = len(audio_urls) - already
    print(f"Audio files: {already} cached, {to_download} to download")

    def download_one(url):
        local_path = url_to_path(url)
        if local_path.exists():
            return url, str(local_path), True
        try:
            with fs.open(url, "rb") as remote:
                with open(local_path, "wb") as local:
                    local.write(remote.read())
            return url, str(local_path), True
        except Exception as e:
            return url, None, False

    if to_download > 0:
        print(f"Downloading {to_download} audio files with {workers} threads...")
        failed = 0
        urls_to_dl = [url for url in audio_urls if not url_to_path(url).exists()]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(download_one, url) for url in urls_to_dl]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
                _, _, success = future.result()
                if not success:
                    failed += 1
        print(f"Download complete: {to_download - failed} ok, {failed} failed")

    # Build map for all URLs
    audio_map = {}
    for url in audio_urls:
        path = url_to_path(url)
        if path.exists():
            audio_map[url] = str(path)
    return audio_map


def extract_gcs_samples(entries: List[dict], audio_map: dict, source_name: str,
                        text_field: str = "source_text_latin",
                        text_field_fallback: str = "text",
                        description_field: str = "description") -> List[dict]:
    """Build sample dicts from GCS metadata + downloaded audio."""
    samples = []
    skipped = 0
    for idx, entry in enumerate(entries):
        audio_url = entry.get("audio", "")
        if audio_url not in audio_map:
            skipped += 1
            continue

        text = entry.get(text_field) or entry.get(text_field_fallback, "")
        if not text or not text.strip():
            skipped += 1
            continue

        samples.append({
            "id": entry.get("id", f"{source_name}_{idx}"),
            "source": source_name,
            "text": text.strip(),
            "description": entry.get(description_field, "neutral"),
            "audio_path": audio_map[audio_url],
        })

    print(f"[{source_name}] Extracted {len(samples)} samples ({skipped} skipped)")
    return samples
