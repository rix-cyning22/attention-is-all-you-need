import torch
import json
from transformer import Transformer

with open("vocabulary.json", "r") as f:
    vocab = json.load(f)


model = Transformer(
    input_vocab_size=len(vocab["word2idx"]),
    output_vocab_size=len(vocab["word2idx"]),
    input_pad_idx=34188,
    output_pad_idx=34188,
    seq_max_len=vocab["seq_max_len"],
    Nx=6,
    heads=8,
    dropout=0.1,
    forward_expansion=4,
    embed_size=64,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load("aiaun-8.pt", map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)


def vectorise_seq(seq):
    cleaned_seq = ""
    for c in seq:
        if c in {".", "?", "!", ","}:
            cleaned_seq += " "
        cleaned_seq += c
    seq = cleaned_seq.split()
    vectorised_seq = [
        (
            vocab["word2idx"][word]
            if word != "" and word in vocab["word2idx"]
            else vocab["oov_idx"]
        )
        for word in seq
    ]
    l = len(vectorised_seq)
    if l > vocab["seq_max_len"]:
        vectorised_seq = vectorised_seq[-vocab["seq_max_len"] :]
    elif l < vocab["seq_max_len"]:
        vectorised_seq = [vocab["oov_idx"]] * (
            vocab["seq_max_len"] - l
        ) + vectorised_seq
    return vectorised_seq


def preprocess_text(vectorised_seq):
    input_tensor = torch.tensor(vectorised_seq).reshape(1, -1).to(device)
    output_seq = [vocab["oov_idx"]] + vectorised_seq[1:]
    output_tensor = torch.tensor(output_seq).reshape(1, -1).to(device)
    return input_tensor, output_tensor


def get_topk_probable_words(pred, k):
    pred_probs = torch.softmax(pred, dim=-1)
    prob, word_idx = torch.topk(pred_probs, k)
    return {
        vocab["idx2word"][str(word_idx[i].item())]: prob[i].item() * 100
        for i in range(k)
    }


def generate_sequence(text, gen_words=15, k=5):
    model.eval()
    text = text.lower()
    vectorised_seq = vectorise_seq(text)
    pred_seq = []
    next_word_probs = []
    with torch.no_grad():
        for _ in range(gen_words):
            input, output = preprocess_text(vectorised_seq)
            pred = model(input, output)[0, -1, :]
            probable_next_words = get_topk_probable_words(pred, k)
            next_word_probs.append(probable_next_words)
            next_word = list(probable_next_words.keys())[0]
            pred_seq.append(next_word)
            vectorised_seq = vectorised_seq[1:] + [vocab["word2idx"][next_word]]
    return {
        "text": text + " " + " ".join(pred_seq),
        "prob_words": next_word_probs,
    }


if __name__ == "__main__":
    seq = input("Enter text: ")
    print(generate_sequence(seq, gen_words=20))
