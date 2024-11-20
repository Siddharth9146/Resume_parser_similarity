import kagglehub

# Download latest version
path = kagglehub.dataset_download("kshitizregmi/jobs-and-job-description")

print("Path to dataset files:", path)