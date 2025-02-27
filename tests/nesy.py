import nnf
import torch
import torchvision
from pathlib import Path
from deepfa.utils import parse_sapienza_to_fa


asset_path = Path(__file__).parent.parent / "assets"

images = torch.stack(
    [
        torchvision.io.decode_image(str(asset_path / "image_{}.png".format(i))).float()
        / 255
        for i in range(1, 4)
    ]
)

symbol_grounder = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, 3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(16, 32, 3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(32, 16, 3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Flatten(),
    torch.nn.Linear(144 + (240 * 2), 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 3),
    torch.nn.Sigmoid(),
)

symbol_grounder.load_state_dict(torch.load(asset_path / "symbol_grounder_weights.pt"))

image_predictions = symbol_grounder(images)

fa = parse_sapienza_to_fa("G((tired | blocked) -> WX(!fast))")

weights = {
    symbol: symbol_probs
    for symbol, symbol_probs in zip(("blocked", "fast", "tired"), image_predictions.T)
}


def labelling_function(var: nnf.Var) -> torch.Tensor:
    return weights[str(var.name)] if var.true else (1 - weights[str(var.name)]).detach()


state_probabilities = fa.forward(
    labelling_function, accumulate=True, accumulate_collapse_accepting=False
)

print(state_probabilities)

mpe = fa.forward(labelling_function, max_propagation=True)

most_probable_assignment = {}
for symbol, weight in weights.items():
    most_probable_assignment[symbol] = (
        torch.autograd.grad(mpe, weight, retain_graph=True)[0] * weight / mpe
    )

print(mpe)
print(most_probable_assignment)
