# Transformer from scratch

The encoder-decoder Transformer from *Attention Is All You Need* (Vaswani et al., 2017), built step by step in PyTorch. Everything is written from basic layers like `nn.Linear`, `nn.Embedding` and `nn.Dropout`, without using `nn.Transformer` or `nn.MultiheadAttention`.

The notebook goes through tokens, embeddings, positional encoding, attention, multi-head attention, masking, the feed-forward layer, and residual connections with layer norm, then puts them together into the full model. At the end there are shape checks and a few tests to make sure the masks work. The model isn't trained yet.

## Running it

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook transformer_from_scratch.ipynb
```
