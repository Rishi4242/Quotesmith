from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
import pandas as pd

app = Flask(__name__)

# Load model and quotes
model = SentenceTransformer("all-MiniLM-L6-v2")
df = pd.read_csv("quotes.csv")
df["embeddings"] = df["quote"].apply(lambda x: model.encode(x, convert_to_tensor=True))

@app.route("/", methods=["GET", "POST"])
def index():
    result = []
    if request.method == "POST":
        user_input = request.form["query"]
        input_embedding = model.encode(user_input, convert_to_tensor=True)

        scores = [float(util.pytorch_cos_sim(input_embedding, quote_emb)[0]) for quote_emb in df["embeddings"]]
        df["score"] = scores
        top_quotes = df.sort_values("score", ascending=False).head(5)[["quote", "author"]]

        result = top_quotes.to_dict(orient="records")

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)