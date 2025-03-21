from huggingface_hub import HfApi, create_repo, login, hf_hub_download
import os
import zipfile
import shutil
from langchain_huggingface import HuggingFaceEmbeddings

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def embedding():
    embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-base')
    
    return embeddings

def authenticate_and_create_repo(token: str, username: str, repo_name: str) -> str:
    login(token)
    api = HfApi()
    repo_id = f"{username}/{repo_name}"
    try:
        create_repo(repo_id, repo_type="dataset", token=token)
        print(f"Đã tạo repository {repo_id} thành công")
    except Exception as e:
        print(f"Lỗi khi tạo repository hoặc repository đã tồn tại: {e}")
    return repo_id

def upload_backup_to_hub(api: HfApi, backup_file: str, repo_id: str, token: str) -> None:
    if not os.path.exists(backup_file):
        print(f"Lỗi: File {backup_file} không tồn tại")
        return

    try:
        api.upload_file(
            path_or_fileobj=backup_file,
            path_in_repo=backup_file,
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        print(f"Đã tải lên thành công file {backup_file} vào repository {repo_id}")
    except Exception as e:
        print(f"Lỗi khi tải file lên: {e}")

def download_and_extract_zip_from_hub(repo_id: str, filename: str, extract_to: str) -> None:
    zip_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def create_backup(source_dir: str, archive_name: str, base_dir: str) -> None:
    shutil.make_archive(archive_name, 'zip', base_dir, source_dir)


if __name__ == "__main__":
    embedding()