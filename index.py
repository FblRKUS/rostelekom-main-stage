import argparse
import sys
import time
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from parser import CodeIndexer
from vector_store import VectorStore

_DEFAULT_GITHUB_BRANCHES = ("main", "master")
_GITHUB_HOSTS = {"github.com", "www.github.com"}


def _parse_github_repo_url(repo_url: str) -> tuple[str, str, str | None]:
    parsed = urllib.parse.urlparse(repo_url.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.netloc.lower() not in _GITHUB_HOSTS
    ):
        raise ValueError("GitHub URL must use https://github.com/{owner}/{repo}")

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError("GitHub URL must include owner and repository name")

    owner = parts[0]
    repo = parts[1].removesuffix(".git")
    preferred_branch = parts[3] if len(parts) >= 4 and parts[2] == "tree" else None
    return owner, repo, preferred_branch


def _download_github_archive(
    owner: str, repo: str, destination: Path, preferred_branch: str | None = None
) -> Path:
    unique_branches: list[str] = []
    for branch in [preferred_branch, *_DEFAULT_GITHUB_BRANCHES]:
        if branch and branch not in unique_branches:
            unique_branches.append(branch)

    last_404: urllib.error.HTTPError | None = None
    for branch in unique_branches:
        archive_url = (
            f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
        )
        archive_path = destination / f"{repo}-{branch}.zip"
        try:
            with urllib.request.urlopen(archive_url) as response:
                archive_path.write_bytes(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                last_404 = exc
                continue
            raise RuntimeError(f"GitHub archive request failed: {archive_url}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Network error while downloading: {archive_url}"
            ) from exc

        try:
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(destination)
        except zipfile.BadZipFile as exc:
            raise RuntimeError("Downloaded archive is not a valid ZIP file") from exc

        expected_root = destination / f"{repo}-{branch}"
        if expected_root.is_dir():
            return expected_root

        candidates = sorted(path for path in destination.iterdir() if path.is_dir())
        if len(candidates) == 1:
            return candidates[0]

        branch_candidates = [
            path for path in candidates if path.name.endswith(f"-{branch}")
        ]
        if branch_candidates:
            return branch_candidates[0]

        raise RuntimeError("Unable to detect extracted repository root folder")

    if last_404 is not None:
        raise ValueError(
            "Could not download repository archive. Tried branches: "
            + ", ".join(unique_branches)
        ) from last_404
    raise RuntimeError("Repository archive download failed")


def index_repository(
    path: str | None = None,
    github: str | None = None,
    persist_path: str | None = None,
) -> str:
    if not path and not github:
        return "Error: Specify a local path or --github URL."

    tmp_dir_obj = None
    if github:
        print(f"Downloading repository {github}...", file=sys.stderr)
        owner, repo, preferred_branch = _parse_github_repo_url(github)
        tmp_dir_obj = tempfile.TemporaryDirectory(prefix="code-indexer-")
        try:
            temp_root = Path(tmp_dir_obj.name)
            dir_path = _download_github_archive(
                owner, repo, temp_root, preferred_branch
            )
        except Exception:
            tmp_dir_obj.cleanup()
            raise
    else:
        if path is None:
            return "Error: path is None"
        dir_path = Path(path).resolve()
        if not dir_path.is_dir():
            return f"Error: {dir_path} is not a valid directory."

    print(f"Starting indexing for {dir_path}...")
    start_time = time.time()

    indexer = CodeIndexer()
    chunks = indexer.scan_directory(dir_path)
    print(f"Parsed {len(chunks)} chunks.")

    store = VectorStore(persist_path=persist_path) if persist_path else VectorStore()
    store.clear()
    store.add_chunks(chunks)

    elapsed = time.time() - start_time

    if tmp_dir_obj:
        tmp_dir_obj.cleanup()

    return f"Successfully indexed {len(chunks)} chunks in {elapsed:.2f} seconds."


def main():
    parser = argparse.ArgumentParser(description="CodeLens RAG Indexer")
    parser.add_argument("path", type=str, nargs="?", help="Local directory to index")
    parser.add_argument("--github", type=str, help="Public GitHub repository URL")
    args = parser.parse_args()

    result = index_repository(path=args.path, github=args.github)
    print(result)
    if result.startswith("Error:"):
        sys.exit(1)


if __name__ == "__main__":
    main()
