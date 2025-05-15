import streamlit as st
import json
import datetime
from model_utils import load_model, predict

st.title("Abbreviation & Long-form Detection (Setup 2.2.3)")

sentence = st.text_area("Enter a biomedical sentence:")
submit = st.button("Predict")

if "model" not in st.session_state:
    st.session_state.model = load_model("model_2.2.3_bigru_crf_biowordvec.pth")
    with open("label2idx.json", "r") as f:
        st.session_state.label_map = json.load(f)

if submit and sentence.strip():
    model = st.session_state.model
    label_map = st.session_state.label_map
    results = predict(model, sentence, label_map)

    st.write("### Predictions:")
    for word, tag in results:
        st.write(f"{word} → {tag}")

    # Log user interaction
    log_entry = {
        "timestamp": str(datetime.datetime.now()),
        "input": sentence,
        "output": results
    }
    with open("log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
