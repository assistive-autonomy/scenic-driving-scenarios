# Serve a model for an infererence request using text-generation inference endpoints.

model=codellama/CodeLlama-7b-Instruct-hf
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.4 --model-id $model