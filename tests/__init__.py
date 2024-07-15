import nnf
import torch
from deepfa.automaton import DeepFA
from deepfa.utils import generate_random_sequence

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
print("acceptance prob: {}".format(acceptance_prob))

accumulated_acceptance_prob = fa.forward(labelling_function, accumulate=True)
print("accumulated acceptance prob: {}".format(accumulated_acceptance_prob))

# The weight of the most probable path that leads to an accepting state
# with the same probs as above. This is the path q1 -> q2 -> q2. So
# transition ~a is True in the first timestep and transition a & b is true
# in the second timestep.
mpe = fa.forward(labelling_function, max_propagation=True)
print("mpe: {}".format(mpe))


# Compute the posterior of symbols given that the sequence is accepted. This
# is easily done via gradients.

weights = {"a": torch.Tensor([0.3, 0.6]), "b": torch.Tensor([0.8, 0.9])}
for value in weights.values():
    value.requires_grad_()


def labelling_function_(var: nnf.Var) -> torch.Tensor:
    # We have to detach the gradient of the negated literals.
    return weights[str(var.name)] if var.true else 1 - weights[str(var.name)].detach()


acceptance_prob = fa.forward(labelling_function_)
posterior_marginals = {}
for symbol, weight in weights.items():
    posterior_marginals[symbol] = (
        torch.autograd.grad(acceptance_prob, weight, retain_graph=True)[0]
        * weight
        / acceptance_prob
    )
print("posterior marginals: {}".format(posterior_marginals))

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

print("most probable assignment: {}".format(most_probable_assignment))

accepting_sequence = generate_random_sequence(fa, sequence_length=3)
print("random accepting sequence: {}".format(accepting_sequence))

rejecting_sequence = generate_random_sequence(fa, accepting=False, sequence_length=3)
print("random rejecting sequence: {}".format(rejecting_sequence))
