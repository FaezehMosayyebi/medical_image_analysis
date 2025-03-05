import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary


def model_analyzer(
    model, lossfunction, optimizer, get_gradient_magnitude: bool, get_layers: bool
):
    """
    Parameters:
               model: The model to analyse
               lossfunction: The loss function in training process
               optimizer: The optimizer in traning process
               get_gradient_magnitude: True If you want to get gradient magnitude figure
               get_layers: True If you want to get layers, parameters, and network size
    """

    # Device
    if torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        print("Analyzer need a GPU device to analize your model!")
        exit(1)

    # Model
    model = model.to(device)

    # Random input and target
    input = torch.randn(9, 1, 32, 32, 32)
    input = input.to(device)
    target = torch.randn(9, 3, 32, 32, 32)
    target = target.to(device)

    if get_gradient_magnitude == True:
        # Run the model
        outputs = model(input)
        loss = lossfunction(outputs, target)
        loss.backward()

        # Store gradient magnitudes
        gradient_magnitudes = []

        num_grad_layers = 0

        # Calculate gradient magnitudes
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient_magnitudes.append(param.grad.norm().item())
                num_grad_layers += 1

        print(f"The number of layers with gradient = {num_grad_layers}")

        # Plot gradient magnitudes
        plt.plot(range(len(gradient_magnitudes)), gradient_magnitudes)
        plt.xlabel("Layers")
        plt.ylabel("Gradient Magnitude")
        plt.title("Gradient Magnitudes over Layers")
        plt.show()

    if get_layers == True:
        summary(model, input[1, :, :, :, :].shape)
