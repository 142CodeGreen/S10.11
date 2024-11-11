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
# S10.11

## Setup

1. Clone the repository:
```
git clone https://github.com/142CodeGreen/S10.7.git
cd S10.7
```

2. Create the Virtual Environment:
```
python -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```
pip install --upgrade -r requirements.txt
```

4. Export API keys. NVIDIA_API_KEY is for NVIDIA NIM, while OpenAI API Key is needed for Nemo Guardrails. 
```
export NVIDIA_API_KEY="your-api-key-here"
echo $NVIDIA_API_KEY

export OPENAI_API_KEY="your-openai-key-here"
echo $OPENAI_API_KEY
```

5. Run the app.py:
```
python3 app.py
```

6. Deactivate virtual environment when finished:
```
deactivate
```

