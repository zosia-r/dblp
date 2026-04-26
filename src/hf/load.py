



def download_file(repo_id: str, filename: str, local_path: str) -> None:
    """
    Download a file from a Hugging Face repository.

    Args:
        repo_id (str): The ID of the repository (e.g., "username/repo").
        filename (str): The name of the file to download.
        local_path (str): The local path where the file should be saved.
    """
    api = HfApi()
    api.download_file(repo_id=repo_id, filename=filename, local_path=local_path)