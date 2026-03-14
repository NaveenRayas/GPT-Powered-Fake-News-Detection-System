from flask import Flask, render_template, request
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)


ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
PARAPHRASE_MODEL = "t5-base"
DEVICE = 0 if torch.cuda.is_available() else -1


classifier = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL, device=DEVICE)
par_tokenizer = AutoTokenizer.from_pretrained(PARAPHRASE_MODEL)
par_model = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASE_MODEL).to("cuda" if DEVICE == 0 else "cpu")


def detect_fake_or_original(text, threshold=0.60):
    candidate_labels = ["original", "fake"]
    res = classifier(text, candidate_labels)
    top_label = res["labels"][0]
    top_score = float(res["scores"][0])
    if top_score < threshold:
        return {"label": "uncertain", "score": top_score}
    return {"label": top_label, "score": top_score}

def paraphrase(text, max_length=128, num_beams=4, num_return_sequences=2):
    device = "cuda" if DEVICE == 0 else "cpu"
    prefix = "paraphrase: "
    input_text = prefix + text.strip().replace("\n", " ")
    inputs = par_tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(device)
    outputs = par_model.generate(
        inputs,
        max_length=max_length,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        early_stopping=True,
        no_repeat_ngram_size=2,
    )
    out_texts = [par_tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=True) for o in outputs]
    return out_texts

# ---- Routes ----
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/description")
def description():
    return render_template("description.html")

@app.route("/fakenews", methods=["GET", "POST"])
def fakenews():
    result = None
    paraphrases = []
    text = ""
    if request.method == "POST":
        text = request.form["text"]
        if text.strip():
            result = detect_fake_or_original(text)
            paraphrases = paraphrase(text)
    return render_template("fakenews.html", result=result, text=text, paraphrases=paraphrases)

if __name__ == "__main__":
    app.run()
