# S1.2

This is a practise of including Nemo Guardrails on top of S1 repository with adjusted code for a simple RAG chatbot that allows upload of file then conduct Q&A using NVIDIA NIM.  It's orchestrated by LlamaIndex and GPU-accelerated Milvus Vector Store, using NVIDIA embeddings. 

## Setup

1. Clone the repository:
```
git clone https://github.com/142CodeGreen/S1.git
cd S1
```

2. Install the required packages:
```
pip install --upgrade -r requirements.txt
```

3. Export API keys. NVIDIA_API_KEY is for NVIDIA NIM, while OpenAI API Key is needed for Nemo Guardrails. 
```
export NVIDIA_API_KEY="your-api-key-here"
echo $NVIDIA_API_KEY

```

4. Run the app.py:
```
python3 app.py
```
