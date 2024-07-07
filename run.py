from torch import nn, optim
import torch
from transformer import Transformer
from datetime import datetime
from transformer import Transformer
import pickle
from matplotlib import pyplot as plt

with open("vocabulary.pkl", "rb") as f:
    vocab_file = pickle.load(f)
vocab, inv_vocab = vocab_file

model = Transformer(
    input_vocab_size=len(vocab),
    output_vocab_size=len(vocab),
    input_pad_idx=vocab["<oov>"],
    output_pad_idx=vocab["<oov>"],
    seq_max_len=15,
    Nx=6,
    heads=8,
    dropout=0.1,
    forward_expansion=4,
    embed_size=64,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load("aiaun.pt", map_location=device)
model.load_state_dict(state_dict)


def train_model(dataloader, num_epochs=5, lr=0.001):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<oov>"])
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    l = len(dataloader)
    for epoch in range(num_epochs):
        start_timestamp = datetime.now()
        losses = []
        loss800 = []
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        for i, (input_batch, output_batch) in enumerate(dataloader):
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)

            optimizer.zero_grad()
            output = model(input_batch, output_batch)
            pred = output.view(-1, output.size(-1))
            target = output_batch.contiguous().view(-1)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if i == 0:
                loss800.append(loss.item())
            if (i + 1) % 800 == 0:
                loss800.append(loss.item())
                curr_timestamp = datetime.now()
                elapsed = (curr_timestamp - start_timestamp).total_seconds()
                eta = elapsed / (i + 1) * (l - i - 1)
                print(
                    f"[{i+1}/{l}] ({((i+1)*100/l):.4f}% done), Loss: {loss.item():.4f}"
                )
                print(f"Elapsed: {elapsed:.4f}s, ETA: {eta:.4f}s")

        avg_loss = sum(losses) / l
        print("_" * 50)
        curr_timestamp = datetime.now()
        elapsed = (curr_timestamp - start_timestamp).total_seconds()
        print(f"Average Loss: {avg_loss:.4f}, Time taken: {elapsed:.4f}s")
        print("=" * 50)
        plt.subplots(figsize=(10, 5))
        plt.plot(losses)
        plt.plot([800 * i for i in range(len(loss800))], loss800)
        plt.legend(["Loss", "Loss800"])
        plt.title(f"Loss, Epoch {epoch+1}")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.savefig(f"loss_epoch{epoch+1}.png")


def preprocess_text(sequence):
    sequence = "".join(
        [s for s in list(sequence.lower()) if s not in {",", ".", ":", ";"}]
    )
    tokenized_seq = sequence.split()
    vectorized_seq = [vocab[s] if s in vocab else vocab["<oov>"] for s in tokenized_seq]
    if len(vectorized_seq) < 15:
        vectorized_seq += [vocab["<oov>"]] * (15 - len(vectorized_seq))
    else:
        vectorized_seq = vectorized_seq[:15]
    return vectorized_seq


def predict(vectorized_seq):
    shifted_seq = vectorized_seq[:-1] + [0]
    input_seq = torch.tensor([vectorized_seq])
    output_seq = torch.tensor([shifted_seq])
    pred = model(input_seq, output_seq)
    return int(torch.argmax(pred[0, -1, :]) + 1)


def generate_sequence(sequence, generation_length=20):
    vectorised = preprocess_text(sequence)
    generated_sequence = []
    for _ in range(generation_length):
        generated_word = predict(vectorised)
        generated_sequence.append(generated_word)
        vectorised = vectorised[1:] + [generated_word]

    return " ".join([inv_vocab[g] for g in generated_sequence])


if __name__ == "__main__":
    seq = input("Enter text: ")
    print(generate_sequence(seq))
