# %% [markdown]
# # Training a GPT model from scratch with Dora the Explorer subtitles
#
# In this notebook, we will train a GPT model from scratch using the Dora the Explorer subtitles dataset. We will use the `PyTorch` library and its `PyTorch Lightning` wrapper to simplify the training process. This is just a demo to show how to train a model, and the training process will be very fast and not very accurate. But it's good for illustration purposes.
# %%
# # Loading the data
# The dataset is a collection of Dora the Explorer subtitles. We will load the data and preprocess it to prepare it for training.
#
# We'll use the OpenSubtitles API to download the subtitles.
# %%
import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import logging
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Set the random seed for reproducibility
torch.manual_seed(42)

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
# Take all files matching subtitles/*.srt
def get_subtitle_files():
    subtitle_files = []
    for root, _, files in os.walk("subtitles"):
        for file in files:
            if file.endswith(".srt"):
                subtitle_files.append(os.path.join(root, file))
    return subtitle_files


# Process the subtitles into plain text
# Use the library `pysrt` to parse the subtitles
def process_subtitle_file(file_path):
    import pysrt

    subs = pysrt.open(file_path)
    text = ""
    for sub in subs:
        text += sub.text + " "

    # Remove all formatting
    # e.g.: <l>Once upon a time, in the magical land of Equestria,</l> -> Once upon a time, in the magical land of Equestria,
    # Do this with regex
    text = re.sub(r"<.*?>", "", text)
    # Remove all newlines
    text = re.sub(r"\n", " ", text)
    # Remove all extra spaces
    text = re.sub(r"\s+", " ", text)
    # Remove all leading and trailing spaces
    text = text.strip()
    # Remove all non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    # Also, sometimes we have ellipses in the subtitles, which are not needed
    # e.g.: "She's gone...! ...Oh no." -> "She's gone! Oh no."
    text = re.sub(r"\.\.\.", "", text)

    # Also, sometimes we have dialogues in the subtitles, which contain -
    # e.g.: "-Mommy? -Winter Wrap Up!" -> "Mommy? Winter Wrap Up!"
    text = re.sub(r"-", "", text)

    # Sometimes we have something like [Captioning sponsored by THE U.S. DEPARTMENT OF EDUCATION and NICKELODEON]
    text = re.sub(r"\[.*?\]", "", text)

    # Remove sequences of 2+ spaces
    text = re.sub(r"\s{2,}", " ", text)

    # Ensure there's a space after punctuation
    text = re.sub(r"([.,!?;:])([^\s])", r"\1 \2", text)

    # Extra rule: take everything saying "Dora" and turn it into "Aldi"
    text = re.sub(r"\bDora\b", "Aldi", text)
    text = re.sub(r"\bDORA\b", "ALDI", text)

    return text


# Download the subtitles from the OpenSubtitles API
def download_subtitles():
    url = "https://www.opensubtitles.org/en/search/subs/mylittlepony"
    response = requests.get(url)
    with open("subtitles.zip", "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile("subtitles.zip", "r") as zip_ref:
        zip_ref.extractall("subtitles")
    os.remove("subtitles.zip")
    logging.info("Subtitles downloaded and extracted.")


# Download the subtitles if they are not already downloaded
if not os.path.exists("subtitles"):
    download_subtitles()

# Load the subtitles files
subtitle_files = get_subtitle_files()

# Process the subtitles files into plain text
subtitles = {}
for file_path in tqdm(subtitle_files):
    text = process_subtitle_file(file_path)
    subtitles[file_path] = text
# %%
# Build a dataset from the subtitles
import datasets

# Create a dataset from the subtitles
dataset = datasets.Dataset.from_dict(
    {
        "text": [text for text in subtitles.values()],
        "filename": [file_path for file_path in subtitles.keys()],
    }
)

# # Temp debugging
# dataset = datasets.Dataset.from_dict(
#     {
#         "text": ["Once upon a time in the land of Equestria"] * 100
#         + ["Once upon a time in the land where dreams come true"] * 100
#         + ["Once upon a time in the land of friendship"] * 100,
#         "filename": ["test"] * 300,
#     }
# )

# # Temp debugging
# dataset = datasets.Dataset.from_dict(
#     {
#         "text": ["Once upon a time in the land of Equestria"] * 100,
#         "filename": ["test"] * 100,
#     }
# )


# %% [markdown]
# # Tokenization
#
# We will train our own tokenizer using the `tokenizers` library. We will use the `ByteLevelBPETokenizer` to train a tokenizer on the subtitles dataset. The tokenizer will be used to encode the text into tokens that can be fed into the model.
# %%
from tokenizers import Tokenizer, models, pre_tokenizers, normalizers, processors
from tokenizers.trainers import BpeTrainer

# Initialize the tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="[Unknown]"))

# Add special tokens: ===text-start===, [Padding], ===unknown===
special_tokens = ["[TextStart]", "[Padding]", "[Unknown]", "[TextEnd]"]
tokenizer.add_special_tokens(special_tokens)

# Preprocessing: remove all special characters, lowercase
tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation(),
    ]
)

tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.NFKC(),
        normalizers.Strip(),
        normalizers.Lowercase(),
    ]
)

# The template is: [TextStart] text
tokenizer.post_processor = processors.TemplateProcessing(
    # single="[TextStart] $0 [TextEnd]",
    single="[TextStart] $0",
    special_tokens=[(t, tokenizer.token_to_id(t)) for t in special_tokens],
)

# Train the tokenizer on the dataset
trainer = BpeTrainer(
    vocab_size=5000,
    min_frequency=2,
    special_tokens=special_tokens,
    limit_alphabet=1000,
    continuing_subword_prefix="##",  # Specify a prefix for continuing subwords
)
tokenizer.train_from_iterator(
    (line for line in dataset["text"]),
    trainer=trainer,
)
# %%
# See how tokenization works
# Encode some text
text = "Aldi the Explorer, take me on an adventure!"
encoded = tokenizer.encode(text)
# Decode the text
decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
print(f"Original text: {text}")
print(f"Encoded text: {encoded.tokens}")
print(f"Decoded text: {decoded}")

# %% [markdown]
# # Training
#
# We will train the model using the `PyTorch` library. We will define a custom training loop using PyTorch components manually. We will use the `AdamW` optimizer and the `CrossEntropyLoss` loss function.
# %%
# We'll write a simple GPT model using elements from PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_size: int,
        query_embedding_size: int,
        key_embedding_size: int,
        value_embedding_size: int,
    ):
        super(SelfAttention, self).__init__()
        assert (
            query_embedding_size == key_embedding_size
        ), "Key and query embedding sizes must match"
        self.query_embedding_size = query_embedding_size
        self.key_embedding_size = key_embedding_size
        self.value_embedding_size = value_embedding_size

        self.W_q = torch.nn.Parameter(torch.rand(embed_size, query_embedding_size))
        self.W_k = torch.nn.Parameter(torch.rand(embed_size, key_embedding_size))
        self.W_v = torch.nn.Parameter(torch.rand(embed_size, value_embedding_size))

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor):
        queries = embeddings @ self.W_q
        keys = embeddings @ self.W_k
        values = embeddings @ self.W_v

        energy = queries @ keys.transpose(-2, -1)

        causal_mask = torch.tril(
            torch.ones(embeddings.size(1), embeddings.size(1), device=energy.device)
        ).bool()
        energy = energy.masked_fill(~causal_mask, value=float("-1e20"))

        expanded_mask = (
            (mask == 0)
            .view(mask.size(0), 1, mask.size(-1))
            .expand(energy.size(0), mask.size(-1), mask.size(-1))
        )

        energy = energy.masked_fill(expanded_mask, value=float("-1e20"))

        epsilon = torch.finfo(torch.float32).eps
        attention = F.softmax(
            energy
            / (
                torch.sqrt(torch.tensor(self.key_embedding_size, dtype=torch.float32))
                + epsilon
            ),
            dim=-1,
        )
        attention_output = attention @ values

        return {"output": attention_output, "attention": attention}


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        query_embedding_size: int,
        key_embedding_size: int,
        value_embedding_size: int,
    ):
        super(MultiHeadSelfAttention, self).__init__()
        assert (
            embed_size % num_heads == 0
        ), "Embedding size must be divisible by number of heads"
        self.embed_size = embed_size
        self.input_embed_size = embed_size // num_heads
        self.num_heads = num_heads

        self.heads = nn.ModuleList(
            [
                SelfAttention(
                    self.input_embed_size,
                    query_embedding_size,
                    key_embedding_size,
                    value_embedding_size,
                )
                for _ in range(num_heads)
            ]
        )
        self.fc_out = torch.nn.Linear(num_heads * value_embedding_size, embed_size)

    def separate_embeddings_into_heads(self, embeddings) -> torch.Tensor:
        return embeddings.view(
            embeddings.size(0),
            embeddings.size(1),
            self.num_heads,
            self.input_embed_size,
        ).transpose(1, 2)

    def concatenate_head_attns_into_single_attn_tensor(
        self, head_attns: torch.Tensor
    ) -> torch.Tensor:
        return head_attns.transpose(1, 2).reshape(
            head_attns.size(0), head_attns.size(2), -1
        )

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor):
        head_inputs = self.separate_embeddings_into_heads(embeddings=embeddings)
        head_outputs = []
        head_attentions = []
        for i, head in enumerate(self.heads):
            head_output_dict = head(head_inputs[:, i, :, :], mask)
            head_output = head_output_dict["output"]
            head_attention = head_output_dict["attention"]
            assert head_output.shape == torch.Size(
                [
                    embeddings.size(0),
                    embeddings.size(1),
                    self.heads[0].value_embedding_size,
                ]
            )
            head_outputs.append(head_output)
            head_attentions.append(head_attention)

        concatenated = self.concatenate_head_attns_into_single_attn_tensor(
            torch.stack(head_outputs, dim=1)
        )

        return {
            "output": self.fc_out(concatenated),
            "attentions": torch.stack(head_attentions, dim=1),
        }


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(
            embed_size,
            heads,
            embed_size // heads,  # We could potentially choose a different value
            embed_size // heads,
            embed_size // heads,
        )
        self.norm1 = nn.LayerNorm(
            embed_size, eps=1e-12
        )  # Increased epsilon for stability
        self.norm2 = nn.LayerNorm(
            embed_size, eps=1e-12
        )  # Increased epsilon for stability
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),  # Changed to GELU for better performance
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

        # Scale factors for residual connections to improve stability
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))

    def forward(self, x, mask):
        # Pre-LayerNorm architecture (more stable)
        normed_x = self.norm1(x)
        attention_output_dict = self.attention(normed_x, mask)
        x_attn = attention_output_dict["output"]
        # Scaled residual connection
        x = x + self.alpha1 * self.dropout(x_attn)

        normed_x = self.norm2(x)
        forward = self.feed_forward(normed_x)
        # Scaled residual connection
        out = x + self.alpha2 * self.dropout(forward)

        return {
            "output": out,
            "attentions": attention_output_dict["attentions"],
        }


class GPT(nn.Module):
    def __init__(
        self,
        embed_size,
        heads,
        num_layers,
        forward_expansion,
        dropout,
        max_length,
        num_classes,
    ):
        super(GPT, self).__init__()
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.max_length = max_length

        # Create token embeddings - these convert token IDs to vectors
        # Think of this as giving each word its own unique representation
        self.embeddings = nn.Embedding(
            num_embeddings=num_classes, embedding_dim=embed_size
        )
        # We're using PyTorch's default initialization for these embeddings
        # This gives each token a good starting representation

        # Add position information using mathematical sine/cosine patterns
        # This helps the model understand word order in a sentence
        self.register_buffer(
            "position_encodings", self._get_sinusoidal_encodings(max_length, embed_size)
        )

        # Rest of the model
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def _get_sinusoidal_encodings(self, max_len, d_model):
        """
        Create position information using sine and cosine waves.
        
        This is a clever mathematical trick that gives each position in a 
        sequence a unique pattern. It helps the model understand word order.

        Args:
            max_len: How many positions we need to encode
            d_model: Size of each position's encoding vector

        Returns:
            A table of position encodings that the model will use
        """
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Create encodings tensor
        encodings = torch.zeros(max_len, d_model)

        # Apply sin to even indices in the array
        encodings[:, 0::2] = torch.sin(position * div_term[: encodings.size(1) // 2])

        # Apply cos to odd indices in the array
        if d_model % 2 == 0:
            encodings[:, 1::2] = torch.cos(
                position * div_term[: encodings.size(1) // 2]
            )
        else:
            encodings[:, 1::2] = torch.cos(
                position * div_term[: encodings.size(1) // 2 - 1]
            )

        return encodings

    def forward(self, x: torch.LongTensor, mask: torch.FloatTensor):
        # Embed the inputs
        x_embeds = self.embeddings(x)

        # Add positional encodings (now using fixed sinusoidal encodings)
        batch_size, seq_len = x.size()
        position_encodings = self.position_encodings[:seq_len, :].unsqueeze(0)

        # Add positional encodings to token embeddings
        x_embeds = x_embeds + position_encodings.to(x_embeds.device)

        attentions = []

        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            transformer_block_output = transformer_block(x_embeds, mask)
            x_embeds = transformer_block_output["output"]
            attentions.append(transformer_block_output["attentions"])

        # Pass through final fully connected layer
        logits = self.fc_out(x_embeds)
        return {"logits": logits, "attentions": torch.stack(attentions, dim=1)}


# %%
embed_size = 120
heads = 4
num_layers = 3
forward_expansion = 4
dropout = 0.1
max_length = 24
num_classes = len(tokenizer.get_vocab())  # Number of tokens in the vocabulary

model = GPT(
    embed_size,
    heads,
    num_layers,
    forward_expansion,
    dropout,
    max_length,
    num_classes,
).to(device)
# Test the GPT model with a dummy input
dummy_input = torch.randint(0, num_classes, (1, max_length)).to(
    device
)  # Random input tensor
dummy_mask = torch.ones(1, max_length).to(device)  # Updated mask tensor
dummy_output = model(dummy_input, dummy_mask)["logits"]  # Forward pass
print(f"Dummy output shape: {dummy_output.shape}")


# %% [markdown]
# # Training Loop
#
# We will define a custom training loop using PyTorch components manually.
# %%
# Define a custom training loop using PyTorch Lightning components manually
import torch.optim as optim
from typing import List


# Initialize the model
model = GPT(
    embed_size,
    heads,
    num_layers,
    forward_expansion,
    dropout,
    max_length,
    num_classes,
).to(device)


class DoraTheExplorerDataset(Dataset):
    """
    This class is used to create a dataset from the Dora the Explorer subtitles. It takes care of tokenization and padding.
    """

    def __init__(self, tokenizer, original_dataset):
        self.tokenizer = tokenizer
        self.original_dataset = original_dataset
        self.prepared_examples, self.prepared_attn_masks = self.create_batches_2(
            original_dataset
        )

    def __len__(self):
        return len(self.prepared_examples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.prepared_examples[idx],
            "attention_mask": self.prepared_attn_masks[idx],
        }

    def create_batches_2(self, original_dataset, NUM_TOKENS_THRESHOLD=5):
        """
        Create batches by taking the original dataset, splitting the examples on '.', and
        returning the resulting batches and attention masks, always padded/trimmed to max_length
        """
        pad_token = self.tokenizer.token_to_id("[Padding]")
        pre_batch = []

        for example in original_dataset:
            sentences = example["text"].split(".")
            for sentence in sentences:
                if sentence.strip():  # Ignore empty sentences
                    encoded = self.tokenizer.encode(sentence.strip())
                    if len(encoded.ids) > NUM_TOKENS_THRESHOLD:
                        pre_batch.append(encoded.ids)

        # Pad or truncate each sequence to max_length
        padded_batch = torch.full(
            (len(pre_batch), max_length), pad_token, dtype=torch.long
        )
        for i, seq in enumerate(pre_batch):
            seq = seq[:max_length]  # Truncate if longer than max_length
            padded_batch[i, : len(seq)] = torch.LongTensor(seq)

        print(f"Shape of padded batches: {padded_batch.shape}")

        # Create an attention mask where 1 indicates valid tokens and 0 indicates padding
        attention_mask = (padded_batch != pad_token).float()

        return padded_batch, attention_mask

    def create_batches(self, original_dataset):
        # pre_batch is a list of LongTensors of different shapes, so to stack them, we need to pad them to the maximum length
        pre_batch = [
            self.tokenizer.encode(example["text"]) for example in original_dataset
        ]
        max_examples_length = max([len(seq.ids) for seq in pre_batch])
        pad_token = tokenizer.token_to_id("[Padding]")
        padded_batch = torch.full(
            (len(pre_batch), max_examples_length), pad_token, dtype=torch.long
        )
        for i, seq in enumerate(pre_batch):
            padded_batch[i, : len(seq.ids)] = torch.LongTensor(seq.ids)

        # Now, separate the padded_batch into chunks of max_length with a sliding window approach
        chunks = padded_batch.unfold(1, max_length, step=16)
        chunks = chunks.reshape((-1, max_length))

        # Create an attention mask where 1 indicates valid tokens and 0 indicates padding
        attention_mask = (chunks != pad_token).float()

        return chunks, attention_mask


# Create the dataset
tokenized_dataset = DoraTheExplorerDataset(tokenizer, dataset)


def collate_fn(x):
    input_ids = [item["input_ids"] for item in x]
    attention_mask = [item["attention_mask"] for item in x]
    return torch.stack(input_ids, dim=0), torch.stack(attention_mask, dim=0)


# Create the DataLoader with the custom function to create batches
dataloader = DataLoader(
    tokenized_dataset,  # Dataset to use
    batch_size=64,  # Smaller batch size for more stable updates
    collate_fn=collate_fn,  # Custom function to create batches
    shuffle=True,  # Shuffle the data
)


# %%
def plot_attentions(attentions: torch.Tensor, tokens: torch.LongTensor):
    # Plot the attention matrices for each layer
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, num_layers, figsize=(3 * num_layers, 3))

    # Stack them (because they're per-layer) and average the batch, layer, & head dims
    averaged_attentions = attentions.squeeze(dim=0).mean(dim=1)

    for layer_idx in range(num_layers):
        ax = axes[layer_idx] if num_layers > 1 else axes
        attention_matrix = averaged_attentions[layer_idx, :, :]

        cax = ax.matshow(attention_matrix.cpu().numpy(), cmap="viridis")
        fig.colorbar(cax, ax=ax)

        ax.set_title(f"Attention Matrix - Layer {layer_idx + 1}", pad=20)
        ax.set_xlabel("Key Positions")
        ax.set_ylabel("Query Positions")

        # Get a list of individual token texts
        token_ids = tokens[0].tolist()
        token_labels = [
            tokenizer.decode([token_id], skip_special_tokens=False)
            for token_id in token_ids
        ]

        # Ensure we have exactly the right number of ticks matching the attention matrix dimensions
        ax.set_xticks(range(len(token_labels[: attention_matrix.size(-1)])))
        ax.set_yticks(range(len(token_labels[: attention_matrix.size(-2)])))

        ax.set_xticklabels(
            token_labels[: attention_matrix.size(-1)],
            rotation=90,
            fontsize=8,
        )
        ax.set_yticklabels(
            token_labels[: attention_matrix.size(-2)],
            fontsize=8,
        )

    plt.tight_layout()
    plt.show()


# Generation with temperature and embedding distance tracking
def generate_text_greedy(
    model, tokenizer, prompt, max_length, should_plot_attentions=False, temperature=0.8
):
    model.eval()
    tokens = tokenizer.encode(prompt, add_special_tokens=False).ids
    tokens = torch.LongTensor(tokens).unsqueeze(0).to(device)

    # Keep track of embedding distances during generation (for informational purposes)
    token_distances = []

    for i in range(tokens.size(-1), max_length):
        with torch.no_grad():
            output = model(tokens, mask=torch.ones(tokens.shape, device=device))

            # For monitoring purposes only - track token embedding distances
            if i % 5 == 0:  # Don't check every token to save time
                token_embeds = model.embeddings.weight.data
                # Sample a subset of tokens to check distances
                idx1 = torch.randint(
                    0, token_embeds.size(0), (100,), device=token_embeds.device
                )
                idx2 = torch.randint(
                    0, token_embeds.size(0), (100,), device=token_embeds.device
                )
                mask = idx1 != idx2
                distances = []
                for j, k in zip(idx1[mask], idx2[mask]):
                    dist = torch.norm(token_embeds[j] - token_embeds[k], p=2).item()
                    distances.append(dist)
                avg_distance = sum(distances) / len(distances) if distances else 0
                token_distances.append(avg_distance)
                logging.info(
                    f"Token embedding average distance at step {i}: {avg_distance:.4f}"
                )

        if should_plot_attentions:
            plot_attentions(output["attentions"], tokens)

        # Get logits and apply temperature
        logits = output["logits"][:, -1, :] / temperature

        # Show top predictions
        top_k = torch.topk(logits, k=5).indices.squeeze().tolist()
        top_k_tokens = [tokenizer.id_to_token(token) for token in top_k]
        top_k_probs = torch.softmax(logits, dim=-1)[0, top_k].tolist()

        logging.info(
            f"Generated so far: {tokenizer.decode(tokens[0].tolist(), skip_special_tokens=False)}"
        )
        logging.info(
            f"Top 5 predictions: {list(zip(top_k_tokens, [f'{p:.4f}' for p in top_k_probs]))}"
        )

        # Sample from the distribution for more diversity
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs[0], num_samples=1)
        tokens = torch.cat((tokens, next_token.unsqueeze(0)), dim=1)

    # Log token distance trend
    if token_distances:
        logging.info(
            f"Token embedding distance trend during generation: {token_distances}"
        )

    return tokenizer.decode(tokens[0].tolist(), skip_special_tokens=False)


def generate_text_beam_search(
    model,
    tokenizer,
    prompt,
    max_length,
    beam_width=5,
    should_plot_attentions=False,
    temperature=1.0,
):
    """
    Generate text using beam search, which considers multiple possible sequences at each step
    and selects the most likely ones to continue expanding.

    Args:
        model: The language model to generate with
        tokenizer: Tokenizer to encode/decode text
        prompt: The initial prompt to start generation from
        max_length: Maximum sequence length
        beam_width: Number of beams (sequences) to track
        should_plot_attentions: Whether to plot attention patterns
        temperature: Temperature for sampling (higher = more random)

    Returns:
        The generated text from the highest scoring beam
    """
    model.eval()

    # Encode the prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=False).ids
    input_ids = torch.LongTensor(tokens).unsqueeze(0).to(device)

    # Initialize beams with the input sequence
    beams = [(input_ids, 0)]  # (sequence, score)

    # Keep track of the best beam for attention visualization
    best_beam_attentions = None

    # Track embedding distances
    token_distances = []

    # Generate tokens up to max_length
    input_len = input_ids.size(-1)
    for i in range(input_len, max_length):
        # Collect all candidate next beams
        all_candidates = []

        # For each current beam
        for beam_tokens, beam_score in beams:
            # Forward pass
            with torch.no_grad():
                output = model(
                    beam_tokens, mask=torch.ones(beam_tokens.shape, device=device)
                )

                # Store attentions for the best beam
                if beam_tokens.equal(beams[0][0]) and should_plot_attentions:
                    best_beam_attentions = output["attentions"]

                # Get logits for the next token
                logits = output["logits"][:, -1, :] / temperature

                # Get probabilities
                probs = torch.softmax(logits, dim=-1)

                # Get top-k token indices and their probabilities
                topk_probs, topk_indices = torch.topk(probs, beam_width)

                # Calculate token embedding distances occasionally (for informational purposes)
                if i % 5 == 0 and beam_tokens.equal(beams[0][0]):
                    token_embeds = model.embeddings.weight.data
                    # Sample a subset of tokens to check distances
                    idx1 = torch.randint(
                        0, token_embeds.size(0), (100,), device=token_embeds.device
                    )
                    idx2 = torch.randint(
                        0, token_embeds.size(0), (100,), device=token_embeds.device
                    )
                    mask = idx1 != idx2
                    distances = []
                    for j, k in zip(idx1[mask], idx2[mask]):
                        dist = torch.norm(token_embeds[j] - token_embeds[k], p=2).item()
                        distances.append(dist)
                    avg_distance = sum(distances) / len(distances) if distances else 0
                    token_distances.append(avg_distance)
                    logging.info(
                        f"Token embedding average distance at step {i}: {avg_distance:.4f}"
                    )

                # Add next token candidates to all_candidates
                for j in range(beam_width):
                    # Get the next token and its log probability
                    next_token_id = topk_indices[0, j].unsqueeze(0).unsqueeze(0)
                    next_token_logprob = torch.log(topk_probs[0, j]).item()

                    # Create new candidate beam
                    new_beam_tokens = torch.cat([beam_tokens, next_token_id], dim=1)
                    new_beam_score = beam_score + next_token_logprob

                    all_candidates.append((new_beam_tokens, new_beam_score))

        # Select top beam_width candidates
        beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        # Show some info about the best beam
        best_beam = beams[0][0]
        if i % 5 == 0 or i == max_length - 1:
            logging.info(
                f"Generated so far: {tokenizer.decode(best_beam[0].tolist(), skip_special_tokens=False)}"
            )

            # Get the probabilities for the top tokens at this step for the best beam
            with torch.no_grad():
                best_output = model(
                    best_beam, mask=torch.ones(best_beam.shape, device=device)
                )
                best_logits = best_output["logits"][:, -1, :] / temperature
                top_k = torch.topk(best_logits, k=5).indices.squeeze().tolist()
                top_k_tokens = [tokenizer.id_to_token(token) for token in top_k]
                top_k_probs = torch.softmax(best_logits, dim=-1)[0, top_k].tolist()
                logging.info(
                    f"Top 5 predictions: {list(zip(top_k_tokens, [f'{p:.4f}' for p in top_k_probs]))}"
                )

    # Get the best beam
    best_beam_tokens = beams[0][0]

    # Plot attentions if requested
    if should_plot_attentions and best_beam_attentions is not None:
        plot_attentions(best_beam_attentions, best_beam_tokens)

    # Log token distance trend
    if token_distances:
        logging.info(
            f"Token embedding distance trend during generation: {token_distances}"
        )

    # Return the decoded text from the best beam
    return tokenizer.decode(best_beam_tokens[0].tolist(), skip_special_tokens=False)


# Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.001)
loss_function = nn.CrossEntropyLoss(
    reduction="none", ignore_index=tokenizer.token_to_id("[Padding]")
)


def calculate_position_encoding_distances(model):
    """
    Calculate the average pairwise Euclidean distance between position encodings.
    This is just for informational purposes as the sinusoidal encodings are fixed.

    Args:
        model: The GPT model with position_encodings buffer

    Returns:
        float: Average pairwise distance between position encodings
    """
    # Get the position encodings
    position_encodings = model.position_encodings

    # Calculate pairwise distances
    total_distance = 0.0
    count = 0

    for i in range(position_encodings.size(0)):
        for j in range(i + 1, position_encodings.size(0)):
            # Calculate Euclidean distance between encodings
            distance = torch.norm(
                position_encodings[i] - position_encodings[j], p=2
            ).item()
            total_distance += distance
            count += 1

    # Return the average distance
    return total_distance / count if count > 0 else 0.0


def calculate_token_embedding_penalty(
    token_embeddings, num_samples=200, scale=0.1, penalty_weight=1.0
):
    """
    Simple placeholder function that no longer applies any penalty.
    With our standard token embeddings, we don't need special constraints.

    Returns:
        torch.Tensor: Zero penalty
    """
    # Return zero penalty (tensor on the same device)
    return torch.tensor(0.0, device=token_embeddings.weight.device)


def calculate_token_embedding_distances(token_embeddings, num_samples=500):
    """
    Calculate the average pairwise Euclidean distance between token embeddings
    using random sampling for efficiency

    Args:
        token_embeddings: Token embedding weight matrix
        num_samples: Number of random token pairs to sample

    Returns:
        float: Average pairwise distance between sampled token embeddings
    """
    # Get the token embeddings weight matrix
    embed_weights = token_embeddings.weight.data
    vocab_size = embed_weights.size(0)

    # Initialize for distance calculation
    total_distance = 0.0

    # For efficiency, sample random pairs rather than computing all pairs
    idx1 = torch.randint(0, vocab_size, (num_samples,), device=embed_weights.device)
    idx2 = torch.randint(0, vocab_size, (num_samples,), device=embed_weights.device)

    # Ensure we don't compare a token to itself
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]

    # Calculate distances
    for i, j in zip(idx1, idx2):
        distance = torch.norm(embed_weights[i] - embed_weights[j], p=2).item()
        total_distance += distance

    # Return the average distance
    return total_distance / len(idx1) if len(idx1) > 0 else 0.0


def normalize_embeddings(model):
    """
    Simple function that doesn't perform any normalization.
    We let the model learn naturally without special adjustments.

    Args:
        model: The GPT model with embeddings
    """
    # No embedding normalization needed with standard initialization
    pass


# %%

# Training loop
num_epochs = 50
first_iteration = True
# Simple hyperparameters
scale = 0.05  # Scale parameter (not actively used with token penalty set to 0)
token_penalty_weight = 0.0  # Zero weight - no token embedding penalty
normalization_interval = 75  # Only used for logging distances now

# Store initial distances for informational purposes
with torch.no_grad():
    initial_token_distance = calculate_token_embedding_distances(model.embeddings)
    initial_pos_distance = calculate_position_encoding_distances(model)
    logging.info(f"Initial token embedding distance: {initial_token_distance:.4f}")
    logging.info(f"Initial position encoding distance: {initial_pos_distance:.4f}")

# No need to apply normalization with our standard embeddings

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_idx, (batch, mask) in enumerate(dataloader):
        # Move batch to device
        batch = batch.to(device)
        mask = mask.to(device)
        assert batch.shape == mask.shape, "Batch and mask shapes must match"
        assert mask.shape[1] <= max_length, "Mask length must not exceed max_length"

        # Print the text of the first & second element
        if first_iteration:
            print(tokenizer.decode(batch[0].tolist(), skip_special_tokens=False))
            print(mask[0].tolist())
            print(f"Shape: {batch.shape}")
            first_iteration = False

        # Forward pass
        outputs = model(x=batch, mask=mask)["logits"]

        # Compute loss - but we need to shift by one because this is causal decoding
        shifted_logits = outputs[:, :-1, :].contiguous()
        shifted_batch = batch[:, 1:].contiguous()
        token_loss = loss_function(
            shifted_logits.view(-1, shifted_logits.size(-1)), shifted_batch.view(-1)
        ).view(shifted_batch.shape)

        # Apply the mask to the loss
        shifted_mask = mask[:, 1:]
        token_loss = token_loss * shifted_mask
        token_loss = token_loss.sum() / shifted_mask.sum()

        # Calculate token embedding penalty only (position encodings are fixed)
        token_embed_penalty = calculate_token_embedding_penalty(
            model.embeddings, scale=scale, penalty_weight=token_penalty_weight
        )

        # Just using the token loss since penalty is set to 0
        loss = token_loss + token_embed_penalty

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()

        # Standard gradient clipping to prevent extreme weight updates
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # No need to normalize embeddings with standard initialization
        # We'll still track distances for information purposes
        with torch.no_grad():
            if batch_idx % normalization_interval == 0:
                token_dist = calculate_token_embedding_distances(model.embeddings)
                if token_dist < 0.5:  # Just log a warning if distances get very small
                    logging.warning(
                        f"INFO: Token embeddings have small distances: {token_dist:.4f}"
                    )

        total_loss += loss.item()

        # Log progress
        if batch_idx % 10 == 0:
            # Calculate current embedding distances for monitoring only
            token_distance = calculate_token_embedding_distances(model.embeddings)

            logging.info(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(dataloader)}], "
                f"Loss: {loss.item():.4f}, Token Loss: {token_loss.item():.4f}, "
                f"Token Distance: {token_distance:.4f}"
            )

    # Log epoch loss
    avg_loss = total_loss / len(dataloader)

    # Calculate token embedding distances for monitoring purposes
    avg_token_distance = calculate_token_embedding_distances(model.embeddings)
    logging.info(
        f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}, "
        f"Token Embedding Distance: {avg_token_distance:.4f}"
    )

    # No normalization needed with our standard approach

# Save the trained model
model_path = "trained_gpt_model.pth"
torch.save(model.state_dict(), model_path)
logging.info(f"Model saved to {model_path}")

# Save the tokenizer as well
tokenizer.save("tokenizer.model")

# %%
# Test at the end with plotting
prompt = "[TextStart] Do you see"
generated_text_greedy = generate_text_greedy(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_length=24,
    should_plot_attentions=False,
)
print(f"Generated text: {generated_text_greedy}")
# %%
