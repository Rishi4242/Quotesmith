import gradio as gr
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

model = SentenceTransformer("all-MiniLM-L6-v2")
df = pd.read_csv("quotes.csv")
df["embeddings"] = df["quote"].apply(lambda x: model.encode(x, convert_to_tensor=True))

def get_quote(user_input):
    input_embedding = model.encode(user_input, convert_to_tensor=True)
    scores = [float(util.pytorch_cos_sim(input_embedding, emb)[0]) for emb in df["embeddings"]]
    df["score"] = scores
    top = df.sort_values("score", ascending=False).head(3)
    return "\n\n".join(f"\"{row['quote']}\" — {row['author']}" for _, row in top.iterrows())

demo = gr.Interface(fn=get_quote, inputs="text", outputs="text", title="QuoteSmith")

demo.launch()
