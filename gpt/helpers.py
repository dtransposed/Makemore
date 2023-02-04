import torch
from typing import Dict, Union, Tuple, List


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.data[idx: idx + self.block_size])
        y = torch.tensor(self.data[idx + 1: idx + 1 + self.block_size])
        return x,y


def encode(inp: str, map_: Dict[str, int]) -> torch.Tensor:
    return [map_[char] for char in inp]


def decode(inp: torch.Tensor, map_: Dict[int, str]) -> str:
    return "".join([map_[tok.item()] for tok in inp[0]])


def load_dataset(path: str, split: float, block_size: int, verbose = True) -> Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset, List[str], Dict[int, str], Dict[str, int]]:
    dataset_str = open(path, 'r').read()
    alphabet = sorted(list(set(dataset_str)))
    char2tok = {char: tok for tok, char in enumerate(alphabet)}
    tok2char = {tok: char for tok, char in enumerate(alphabet)}
    dataset = encode(dataset_str, char2tok)

    split_index = int(len(dataset) * split)
    train_set, val_set = dataset[:split_index], dataset[split_index:]
    if verbose:
        print(f"Dataset has {len(dataset_str)} characters and {len(alphabet)} unique characters")
        print(f"The alphabet is: {''.join(alphabet)}")
        print(f"Size of training set: {len(train_set)}, size of validation set: {len(val_set)}")

    return TextDataset(train_set, block_size), TextDataset(val_set, block_size), alphabet, tok2char, char2tok
