import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

TOKEN = os.getenv("HF_TOKEN")
if not TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it to your Hugging Face API token.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload local files to a Hugging Face repository.")
    parser.add_argument("--repo-id", required=True, help="Target repository, for example: username/repo-name")
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="Repository type on Hugging Face Hub.",
    )
    parser.add_argument("--file", dest="file_path", help="Local file path to upload.")
    parser.add_argument("--folder", dest="folder_path", help="Local folder to upload recursively.")
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Path in target repository. Required for --file; optional for --folder (defaults to root).",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="HF token. Defaults to HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch, tag, or commit to upload to (default: main).",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload via CLI uploader",
        help="Commit message for the upload operation.",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create repository if it does not exist.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="When used with --create-repo, create as private repository.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if bool(args.file_path) == bool(args.folder_path):
        raise ValueError("Provide exactly one of --file or --folder.")

    if args.file_path and not args.path_in_repo:
        raise ValueError("--path-in-repo is required when using --file.")

    if args.file_path and not Path(args.file_path).is_file():
        raise ValueError(f"File does not exist: {args.file_path}")

    if args.folder_path and not Path(args.folder_path).is_dir():
        raise ValueError(f"Folder does not exist: {args.folder_path}")

    if not args.token or TOKEN == "":
        raise ValueError("HF token must be provided via --token or HF_TOKEN environment variable.")


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    api = HfApi(token=args.token if args.token else TOKEN)

    if args.create_repo:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            private=args.private,
            exist_ok=True,
        )

    if args.file_path:
        result = api.upload_file(
            path_or_fileobj=args.file_path,
            path_in_repo=args.path_in_repo,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            commit_message=args.commit_message,
        )
        print(result)
        return

    result = api.upload_folder(
        folder_path=args.folder_path,
        path_in_repo=args.path_in_repo or "",
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        commit_message=args.commit_message,
    )
    print(result)


if __name__ == "__main__":
    main()
