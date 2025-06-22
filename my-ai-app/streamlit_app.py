import streamlit as st
from transformers import pipeline

st.title("🧠 GPT-2 Text Generator (Offline)")
st.subheader("Enter a prompt and let GPT-2 complete it!")

# Load model only once
@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="gpt2")

generator = load_generator()

# Prompt input
query = st.text_input("Type your prompt below:")

# Generate text
if query:
    with st.spinner("Generating..."):
        output = generator(
            query,
            max_length=60,              # Keeps output concise
            temperature=0.7,            # Controls creativity (lower is more focused)
            top_k=50,                   # Limits randomness
            repetition_penalty=1.2      # Avoids repeating phrases
        )
        st.success("Generated Text:")
        st.write(output[0]['generated_text'])
