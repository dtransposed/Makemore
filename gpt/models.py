import torch
from torch.nn import functional as F


class FeedForward(torch.nn.Module):
    def __init__(self, embedding_size: int, dropout: float = 0.2):
        super().__init__()
        self.ff1 = torch.nn.Linear(embedding_size, embedding_size * 4)
        self.ff2 = torch.nn.Linear(embedding_size * 4, embedding_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.ff2(F.relu(self.ff1(x))))


class Block(torch.nn.Module):
    def __init__(self, n_embedd: int, sequence_len: int, num_heads: int):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_size=n_embedd,
                                                       sequence_len=sequence_len,
                                                       num_heads=num_heads)
        # add norm layer
        self.ln1 = torch.nn.LayerNorm(n_embedd)
        # add feed forward layer
        self.ff = FeedForward(n_embedd)
        # add second norm layer
        self.ln2 = torch.nn.LayerNorm(n_embedd)

    def forward(self, x):
        x = x + self.multi_head_attention(x)
        x = self.ln1(x)
        x = x + self.ff(x)
        x = self.ln2(x)
        return x


class GPT(torch.nn.Module):
    def __init__(self, vocab_size: int, sequence_len: int, n_embedd: int, n_blocks: int, n_heads: int):
        super().__init__()
        self.sequence_len = sequence_len
        self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embedd)
        self.position_embedding_table = torch.nn.Embedding(sequence_len, n_embedd)
        self.pred_head = torch.nn.Linear(n_embedd, vocab_size)
        self.blocks = torch.nn.ModuleList([Block(n_embedd, sequence_len, n_heads) for _ in range(n_blocks)])
        self.ln = torch.nn.LayerNorm(n_embedd)

    def forward(self, x):
        B, T = x.size()
        token_embedding = self.token_embedding_table(x)  # (B, T, E)
        position_embedding = self.position_embedding_table(torch.arange(T))  # (T, E)
        x = token_embedding + position_embedding  # (B, T, E)
        for block in self.blocks:
            x = block(x)  # (B, T, E)
        x = self.ln(x)  # (B, T, E)
        logits = self.pred_head(x)  # B, T, V
        return logits

    @torch.no_grad()
    def generate(self, x, max_len: int):
        for _ in range(max_len):
            x_cond = x[:, -self.sequence_len:]
            logits = self.forward(x_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            x = torch.cat([x, idx_next], dim=1)
        return x


class MultiHeadAttention(torch.nn.Module):
    """
    Multiple heads of self-attention running in parallel.
    """

    def __init__(self, embedding_size: int, sequence_len: int, num_heads: int, dropout: float = 0.2):
        super().__init__()
        head_size = embedding_size // num_heads
        assert embedding_size % num_heads == 0, "Embedding size must be divisible by the number of heads"
        self.heads = torch.nn.ModuleList(
            [SelfAttention(embedding_size, sequence_len, head_size) for _ in range(num_heads)])
        self.linear = torch.nn.Linear(embedding_size, embedding_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.linear(x))


class SelfAttention(torch.nn.Module):
    def __init__(self, embedding_size: int, sequence_len: int, head_size: int, dropout: float = 0.2,
                 masking: bool = True):
        """
        :param embedding_size: The size of the embedding vector
        :param sequence_len: The length of the input sequence
        :param head_size: The number of heads to use for the attention

        """
        super().__init__()
        self.key_network = torch.nn.Linear(embedding_size, head_size, bias=False)
        self.query_network = torch.nn.Linear(embedding_size, head_size, bias=False)
        self.value_network = torch.nn.Linear(embedding_size, head_size, bias=False)
        self.head_size = head_size
        self.masking = masking
        self.register_buffer('mask', torch.tril(torch.ones(sequence_len, sequence_len)))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (batch_size, sequence_length, embedding_size)
        B, T, E = x.size()
        keys = self.key_network(x)  # (batch_size, sequence_length, head_size)
        queries = self.query_network(x)  # (batch_size, sequence_length, head_size)
        # keys and queries interact with each other to produce
        # affinity scores between each token in a sequence
        affinities = queries @ keys.transpose(-1, -2) * self.head_size ** (
            -.5)  # (batch_size, sequence_length, sequence_length)
        if self.masking:
            # built masks to prevent future tokens to communicate
            # with previous tokens
            affinities = affinities.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attention = torch.softmax(affinities, dim=-1)  # (batch_size, sequence_length, sequence_length)
        attention = self.dropout(attention)
        value = self.value_network(x)  # (batch_size, sequence_length, head_size)
        return attention @ value  # (batch_size, sequence_length, head_size)
