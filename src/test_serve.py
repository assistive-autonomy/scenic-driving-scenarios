from huggingface_hub import InferenceClient

client = InferenceClient(model="http://goya:8080")
output = client.text_generation(prompt="Write a code for snake game")

print(output)