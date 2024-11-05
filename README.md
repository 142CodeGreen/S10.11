---
title: S1.2
emoji: ⚡
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.4.0
app_file: app.py
pinned: false
---

# S1.2

This is a practise of including Nemo Guardrails on top of S1 repository with adjusted code. It's a simple RAG chatbot that allows upload of file then conduct Q&A using NVIDIA NIM.  The RAG orchestration is via LlamaIndex and GPU-accelerated Milvus Vector Store, using NVIDIA embeddings. 

This workbook includes indexing for loaded document in the doc_loader.py file.

## Setup

1. Clone the repository:
```
git clone https://github.com/142CodeGreen/S1.2.git
cd S1.2
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
