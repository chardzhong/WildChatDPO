#Uploads model folder created by OpenInstruct

from huggingface_hub import HfApi, create_repo
api = HfApi()
name = "Qwen2.5-7b-stock-DPO-Completeness-MinChosen8-MinDelta6"
repo = "chardizard/"+name
folder = "/scratch/rhz2020/DPO/open-instruct/"+name
create_repo(repo)
api.upload_folder(
    folder_path=folder,
    repo_id=repo,
    repo_type="model",
)
