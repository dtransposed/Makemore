from helpers import encode, load_dataset
import torch
import argparse

torch.manual_seed(69)
def main(args: argparse.Namespace):
    train_set, val_set, alphabet, tok2char, char2tok = load_dataset(args.dataset_path, 0, args.block_size, verbose = False)
    model = torch.load(args.model_path).module
    model.eval()
    model.to(args.device)
    print('Enter a sentence that will guide the topic of the podcast:')
    x = input()
    model.generate(encode(x, char2tok), args.generate_length, tok2char=tok2char)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--model_path', type=str, default='model_gpt_6.pt')
    parser.add_argument('--generate_length', type=int, default=2000)
    parser.add_argument('--dataset_path', type=str, default='transcript_lex.txt')
    args = parser.parse_args()
    main(args)