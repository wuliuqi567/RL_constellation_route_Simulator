from .mlp import Basic_Identical, Basic_MLP
from .cnn import Basic_CNN, AC_CNN_Atari
from .rnn import Basic_RNN
from .gnn import Basic_GAT
from .mpnn import Basic_MPNN, MultiRound_MPNN

REGISTRY_Representation = {
    "Basic_Identical": Basic_Identical,
    "Basic_MLP": Basic_MLP,
    "Basic_CNN": Basic_CNN,
    "AC_CNN_Atari": AC_CNN_Atari,
    "Basic_RNN": Basic_RNN,
    "Basic_GAT": Basic_GAT,
    "Basic_MPNN": Basic_MPNN,
    "MultiRound_MPNN": MultiRound_MPNN
}

__all__ = [
    "REGISTRY_Representation",
    "Basic_Identical", "Basic_MLP",
    "Basic_CNN", "AC_CNN_Atari",
    "Basic_RNN",
    "Basic_GAT",
    "Basic_MPNN",
    "MultiRound_MPNN"
]
