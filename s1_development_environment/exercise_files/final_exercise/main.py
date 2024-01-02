import click
import torch
from model import MyAwesomeModel
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_loader, _ = mnist()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = 10
    steps = 0

    train_losses = []
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            model.train()
            optimizer.zero_grad()
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        else:
            epoch_loss = running_loss/len(train_loader)
            print(f"Training loss: {epoch_loss}")
            train_losses.append(epoch_loss)
    torch.save(model, "./output/checkpoint.pth")
    plt.plot(train_losses, [i for i in range(epochs)])
    plt.savefig("./output/training_losses.png")

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_loader = mnist()


    with torch.no_grad():
        model.eval()
        accuracy = []
        for images, labels in test_loader:
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy.append(torch.mean(equals.type(torch.FloatTensor)).item())
        print(f'Accuracy: {np.mean(accuracy)*100}%')


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()