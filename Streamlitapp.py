import streamlit as st
from transformers import pipeline

# Streamlit App UI
st.set_page_config(page_title="Sentiment & Intent Classifier", layout="centered")
st.title("Customer Sentence Classifier")
st.write("Analyze sentiment and intent from customer text using a zero-shot LLM.")

# Initialize the zero-shot classification pipeline
@st.cache_resource(show_spinner="Loading model...")
def get_classifier():
    return pipeline(
        "zero-shot-classification",
        model=r"local_model",
        tokenizer=r"local_model",
        device = -1)

classifier = get_classifier()

# Define candidate labels
sentiment_labels = ['Positive', 'Negative', 'Neutral']
intent_labels = ['Refund Request', 'Complaint', 'Information Request', 'Cancel Account', 'Feedback']

# Function to analyze a sentence
def analyze_sentence(input_sentence):
    sentiment_result = classifier(input_sentence, candidate_labels=sentiment_labels)
    sentiment = sentiment_result['labels'][0]

    intent_result = classifier(input_sentence, candidate_labels=intent_labels)
    top_intents = intent_result['labels'][:2]

    return sentiment, top_intents

# Streamlit App UI Cont..
# Input method
input_method = st.radio("Choose input method:", ("Manual Entry", "Upload .txt File"))

sentences = []

if input_method == "Manual Entry":
    user_input = st.text_area("Enter a sentence:", height=150)
    if user_input.strip():
        sentences.append(user_input.strip())

else:
    uploaded_file = st.file_uploader("Upload a .txt file (one sentence per line)", type="txt")
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        sentences = [line.strip() for line in content.splitlines() if line.strip()]

# Button to trigger classification
if st.button("Classify"):
    if not sentences:
        st.warning("Please enter or upload at least one sentence.")
    else:
        st.success(f" Analyzing {len(sentences)} sentence(s)...")

        for i, sentence in enumerate(sentences, 1):
            sentiment, top_intents = analyze_sentence(sentence)

            st.write(f"\nResult {i}")
            st.markdown(f"**Input:** {sentence}")
            st.markdown(f"**Sentiment:** {sentiment}")
            if sentiment == "Negative":
             st.markdown(f"**Top Intents:** {', '.join(top_intents)}")
            else:
             st.markdown(f"**Top Intents:** {top_intents[0]}")
            st.markdown("---")
        st.markdown(f"\n**Total Analyzed Sentences:** {len(sentences)}")
        if st.button("Run Again"):
            st.experimental_rerun()
