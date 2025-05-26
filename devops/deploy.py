import os
import subprocess
model_name = None

# ==== CONFIGURATION ====
repo_url = "https://github.com/yourusername/your-repo.git"
repo_name = "your-repo"  # Folder name the repo will clone into
docker_image = "model-api-image"
docker_container = "model-api-container"
api_port = 8000  # Change to match your model_api.py port

# ==== STEP 1: Create Dockerfile ====
dockerfile = f"""
FROM python:3.10-slim

# Install Git
RUN apt-get update && apt-get install -y git

# Clone the latest repo
RUN git clone {repo_url} /app

WORKDIR /app

# Install Python requirements
RUN pip install --upgrade pip \\
 && pip install -r requirements.txt

# Expose API port
EXPOSE {api_port}

# Run the API
CMD ["python", "{model_name}"]
"""

with open("Dockerfile", "w") as f:
    f.write(dockerfile.strip())

# ==== STEP 2: Build Docker Image ====
subprocess.run(["docker", "build", "-t", docker_image, "."], check=True)

# ==== STEP 3: Stop and Remove Old Container (if exists) ====
subprocess.run(["docker", "rm", "-f", docker_container], check=False)

# ==== STEP 4: Run Docker Container ====
subprocess.run([
    "docker", "run", "-d",
    "--name", docker_container,
    "-p", f"{api_port}:{api_port}",
    docker_image
], check=True)

print(f"\n✅ Docker container '{docker_container}' running at http://localhost:{api_port}")
