from torch.nn import functional as F
from models import GPT
from helpers import decode, load_dataset
import torch
import argparse
import numpy as np
import wandb
import tqdm


def main(args: argparse.Namespace):
    wandb.init(project="gpt")
    train_set, val_set, alphabet, tok2char, _ = load_dataset(args.dataset_path, args.split, args.block_size)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = GPT(vocab_size=len(alphabet),
                sequence_len=args.block_size,
                n_embedd=args.embedding_dim,
                n_blocks=args.num_blocks,
                n_heads=args.num_heads,
                device=args.device)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)

    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        loss_ = []
        val_loss_ = []
        for iter, (x, y) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            x = x.to(args.device)
            y = y.to(args.device)
            model.train()
            logits = model(x)
            B, T, C = logits.size()
            loss = F.cross_entropy(logits.view(B * T, C), y.view(-1))
            loss_.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for iter, (x, y) in enumerate(val_loader):
            x = x.to(args.device)
            y = y.to(args.device)
            model.eval()
            logits = model(x)
            B, T, C = logits.size()
            loss = F.cross_entropy(logits.view(B * T, C), y.view(-1))
            val_loss_.append(loss.item())

        if epoch % args.print_frequency == 0:
            model.eval()
            model_path = f"{args.model_path}_{epoch}.pt"
            generated_tokens = model.module.generate(torch.zeros(1, 1, dtype=torch.long, device=args.device),
                                                     args.generate_length)
            print(f"Epoch {epoch}, train_loss: {np.mean(loss_):.3f}, val_loss: {np.mean(val_loss_):.3f}")
            generated_text = decode(generated_tokens, tok2char)
            print(generated_text)
            with open('generated.txt', 'a') as f:
                f.write(generated_text)
            torch.save(model, model_path)
            print(f'Saved model: {model_path}')
            print('-----')

        wandb.log({"train_loss": np.mean(loss_), "val_loss": np.mean(val_loss_)})
        torch.save(model, args.model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--num_blocks', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dataset_path', type=str, default='transcript_lex.txt')
    parser.add_argument('--model_path', type=str, default='model_gpt')
    parser.add_argument('--generate_length', type=int, default=200)
    parser.add_argument('--print_frequency', type=int, default=1)
    args = parser.parse_args()
    main(args)
