# A bunch of function which may be helpful to someone somewhen.
import nnf
import torch
from deepfa.automaton import DeepFA


def generate_random_sequence(
    deepfa: DeepFA, accepting: bool = True, sequence_length: int = 5
) -> dict[str, list[bool]]:
    # Generates a random sequence from the automaton.
    # :param deepfa: The deepFA from whose language
    #   to create the random sequence.
    # :param accepting: Whether to creating a sequence
    #   that is accepted by the automaton or not.
    # :param sequence_length: The length of the generated
    #   sequence.
    # :return a dictionary with values boolean lists specifying
    #   with entry list[i] specifying whether the symbol (key) is
    #   true in timestep i in the generated sequence.

    weights = {symbol: torch.rand(sequence_length) for symbol in deepfa.symbols}
    for weight in weights.values():
        weight.requires_grad_()

    def labelling_function(var: nnf.Var) -> torch.Tensor:
        return (
            weights[str(var.name)] if var.true else 1 - weights[str(var.name)].detach()
        )

    mpe = deepfa.forward(
        labelling_function, max_propagation=True, return_accepting=accepting
    )

    most_probable_assignment = {
        symbol: (
            (torch.autograd.grad(mpe, weight, retain_graph=True)[0] * weight / mpe)
            > 0.5
        ).tolist()
        for symbol, weight in weights.items()
    }
    return most_probable_assignment
