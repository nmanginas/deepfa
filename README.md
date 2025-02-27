# Description

DeepFA is a small libary to integate symbolic finite automata with 
knowledge compilation techniques to allow for a clean integration 
between symbolic automata and neural networks

# Installation

With poetry:
```poetry add git+ssh://git@github.com:nmanginas/deepfa.git```

You will also need the [dsharp](https://github.com/QuMuLab/dsharp) system. 
You need to make it and preferably put it on your path altough there 
are options to pass the real path of dsharp in which case you will not 
need to add it to your path.

# Usage

The package defines an easy to use low level API for performing operations
on DeepFA

```python
import nnf
import torch
from deepfa.automaton import DeepFA

a, b = nnf.Var("a"), nnf.Var("b")

transitions = {0: {0: a, 1: ~a}, 1: {1: a & b, 0: (a & b).negate()}}
fa = DeepFA(transitions, 0, {1})
fa.dot().view()

weights = {"a": torch.Tensor([0.3, 0.6]), "b": torch.Tensor([0.8, 0.9])}

def labelling_function(var: nnf.Var) -> torch.Tensor:
    return weights[str(var.name)] if var.true else 1 - weights[str(var.name)]

# The probability of accepting a sequence where a is true in timestep 1 with
# probability 0.3 and with probability 0.6 in timestep 2. For b the probs are
# 0.8 and 0.9 respectivelly.
acceptance_prob = fa.forward(labelling_function)

# The weight of the most probable path that leads to an accepting state
# with the same probs as above. This is the path q1 -> q2 -> q2. So
# transition ~a is True in the first timestep and transition a & b is true
# in the second timestep.
mpe = fa.forward(labelling_function, max_propagation=True)

# Compute the posterior of symbols given that the sequence is accepted. This
# is easily done via gradients.

weights = {"a": torch.Tensor([0.3, 0.6]), "b": torch.Tensor([0.8, 0.9])}
for value in weights.values():
    value.requires_grad_()

def labelling_function_(var: nnf.Var) -> torch.Tensor:
    # We have to detach the gradient of the negated literals.
    return (
        weights[str(var.name)] if var.true else 1 - weights[str(var.name)].detach()
    )

acceptance_prob = fa.forward(labelling_function_)
posterior_marginals = {}
for symbol, weight in weights.items():
    posterior_marginals[symbol] = (
        torch.autograd.grad(acceptance_prob, weight, retain_graph=True)[0]
        * weight
        / acceptance_prob
    )

# Computing the most probable assignment given that the sequence
# is accepted is also trivial via derivatives. The task is to uncover
# the path that leads to the MPE. Usually this is done via backtracking
# but gradient work just as well and are much simpler since the derivative
# of max is essentially 1 for the element which would be chosen in backtracking
# and is zero otherwise.

mpe = fa.forward(labelling_function_, max_propagation=True)

most_probable_assignment = {}
for symbol, weight in weights.items():
    most_probable_assignment[symbol] = (
        torch.autograd.grad(mpe, weight, retain_graph=True)[0] * weight / mpe
    )
```

Here is an example for categorical variables which require a bit of 
special handling.

```python
import nnf
import torch
from deepfa.automaton import DeepFA

# Define 4 symbols the first 3 of which are categorical.
# This means exactly one of them will be true in each timestep
a, b, c, d = map(nnf.Var, ("a", "b", "c", "d"))

# We must define a constraint that tells the compiled circuits
# thate a, b and c are indeed a categorical group.
categorical_constraint = (a | b | c) & (~a | ~b) & (~a | ~c) & (~b | ~c)


transitions = {
    0: {
        0: (a | b) & d & categorical_constraint,
        1: ((a | b) & d).negate() & categorical_constraint,
    },
    1: {0: ~c & categorical_constraint, 1: c & categorical_constraint},
}

deepfa = DeepFA(transitions, 0, {1})

# Observe that for each timestep the sum of the probabilities of
# a, b and c must be 1 since the define a categorical distribution.
weights = {
    "a": torch.Tensor([0.2, 0.2]),
    "b": torch.Tensor([0.7, 0.5]),
    "c": torch.Tensor([0.1, 0.3]),
    "d": torch.Tensor([0.4, 0.7]),
}


def labelling_function(var: nnf.Var) -> torch.Tensor:
    # This is very much like the standard labelling function
    # introduced above but we always give the value 1 for
    # negative literals. It's just the way it is.
    if str(var.name) in ("a", "b", "c"):
        return (
            weights[str(var.name)]
            if var.true
            else torch.ones_like(weights[str(var.name)])
        )

    return weights[str(var.name)] if var.true else 1 - weights[str(var.name)]


# Compute the actual probability
acceptance_prob = deepfa.forward(labelling_function)
```

Specifying an automaton via (P)LTLf. This is just for 
convenience. The integration is done with LTLf2DFA which 
is perhaps the least scalable (P)LTLf compiler. It is based
on MONA so you will need to install it with 
```apt install mona``` on Debian based systems.

```python
from deepfa.utils import parse_sapienza_to_fa

fa = parse_sapienza_to_fa("G((tired | blocked) -> WX(!fast))")
fa.dot().view()
```

# Neuro-symbolic interface

This package also allows for easy integration of symbolic automata
and neural networks. We showcase this on a simple example.

![](assets/image_1.png) ![](assets/image_2.png) ![](assets/image_3.png)

This sequence of three images needs to be classified with a symbolic 
automaton. For each image three symbols must be extracted. From top to 
bottom, these are; whether the road is blocked, whether the car is going fast 
and whether the driver is tired. Each is depicted by an emoji. 
For the middle image the correct labeling is {blocked: 1, fast: 0, tired: 0} (again
read from top to bottom)

A neural network is used to bridge between the complex image
representation and the symbolic input expected by the DFA. To follow 
this example you should install ```torchvision``` which is not in 
the dependencies of the ```deepfa``` package.

```python
import torch
import torchvision

images = torch.stack(
    [
        torchvision.io.decode_image("assets/image_{}.png".format(i)).float()
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

symbol_grounder.load_state_dict(torch.load("assets/symbol_grounder_weights.pt"))

image_predictions = symbol_grounder(images)

print(image_predictions)
```

For the second image the predicted probabilities are: 
{blocked: 0.93, fast: 0, tired: 0.57}. We can now use 
these neural network predictions to compute the probability 
that our symbolic pattern is satisfied by the image sequence. 

```python
from deepfa.utils import parse_sapienza_to_fa
fa = parse_sapienza_to_fa("G((tired | blocked) -> WX(!fast))")
```

This will convert the temporal logic formula to the automaton:

<img title="" src="assets/fa.png" alt="" width="260" data-align="center">

```python
weights = {
    symbol: symbol_probs
    for symbol, symbol_probs in zip(("blocked", "fast", "tired"), image_predictions.T)
}


def labelling_function(var: nnf.Var) -> torch.Tensor:
    return weights[str(var.name)] if var.true else 1 - weights[str(var.name)]


state_probabilities = fa.forward(
    labelling_function, accumulate=True, accumulate_collapse_accepting=False
)

print(state_probabilities)
```

This will show the probability of being in each of the automaton states for the three
timesteps in the sequence. Dropping ```accumulate``` arguments which keep the value
of probabilities for all timesteps would result in the probability of accepting the
sequence which is this case is: ```tensor([0.9848], grad_fn=<SumBackward1>)```. 
Importantly this carries a gradient which can be used to train the system in end to end fashion. 

# Explanation generation.

Let's consider however what is the most possible world in which the sequence is
accepted:

```python
mpe = fa.forward(labelling_function, max_propagation=True)

most_probable_assignment = {}
for symbol, weight in weights.items():
    most_probable_assignment[symbol] = (
        torch.autograd.grad(mpe, weight, retain_graph=True)[0] * weight / mpe
    )

print(mpe)
print(most_probable_assignment)
```

The system finds the most probable explanation for accepting the sequence to be:

| timestep | blocked | fast | tired |
| -------- | ------- | ---- | ----- |
| 1        | 1       | 1    | 1     |
| 2        | 1       | 0    | 1     |
| 3        | 0       | 0    | 0     |

with a probability of 0.16. Interestingly this is also the correct labelling for the image sequence.
