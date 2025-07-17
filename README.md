# Sentiment & Intent Classifier (LLM-Based)

This is a Streamlit web app that uses a Hugging Face transformer model (`facebook/bart-large-mnli`) for zero-shot classification. It predicts sentiment and customer intent from input text.

## Features

- Zero-shot classification using Hugging Face Transformers
- Predicts sentiment: Positive, Negative, Neutral
- Predicts intent: Refund Request, Complaint, Cancel Account, etc.
- Accepts both manual text input and file uploads

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Yousif-AlEshari/Sentiment-Analysis-LLM.git
cd Sentiment-Analysis-LLM
pip install -r requirements.txt
```
## Run the Model
In Command Prompt, navigate to the directory where you pulled the repository and run:
streamlit run Streamlitapp.py