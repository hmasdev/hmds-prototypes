import copy
from typing import Callable
import torch
from torch import nn
from .tocrh_utils import DeviceCheckMixin


class KAN(nn.Module, DeviceCheckMixin):
    """A KAN module

    Kolmogorov-Arnold representation network (KAN) is a neural network that represents a function as a sum of functions of one variable.

    Kolmogorov-Arnold representation theorem states that any function :math:`f_d: [0, 1]^{n_{\\text{in}}} \\to \\mathbb{R}^{n_{\\text{out}}}` is representable as a sum of functions of one variable:

    .. math::

        f_d(x_1, \\cdots, x_{n_\\text{in}}) = \\sum_{q=1}^{2n_\\text{in}+1} \\Phi_q\\left(\\sum_{p=1}^{n_\\text{in}} \\phi_{q,p}(x_p)\\right)

    where :math:`\\Phi_q: \\mathbb{R} \\to \\mathbb{R}^{n_{\\text{out}}}` and :math:`\\phi_{q,p}: [0, 1] \\to \\mathbb{R}`.

    Parameters
    ----------
    n_in : int
        The number of input columns

    n_out : int
        The number of output columns

    base_1d_module : torch.nn.Module
        :math:`\\phi_{q,p}` and :math:`\\Phi_q` in the above equation

    n_hiddens : int, optional
        The number of hidden layers, by default None
        If None, set to 2 * n_in + 1.

    squeeze_1d_output : bool, optional
        Whether to squeeze the output when n_out is 1, by default True
        When n_out is 1, if True, the output is squeezed to (N,), otherwise (N, 1).

    Methods
    -------
    forward(x)
        Forward pass

    References
    ----------
    .. [1] KAN: Kolmogorov-Arnold Networks: https://arxiv.org/abs/2404.19756v2
    .. [2] Gaussian Process Kolmogorov-Arnold Networks: https://arxiv.org/abs/2407.18397
    """  # noqa

    def __init__(
        self,
        n_in: int,
        n_out: int,
        base_1d_module: nn.Module | Callable[[], nn.Module],
        n_hiddens: int | None = None,
        squeeze_1d_output: bool = True,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hiddens = n_hiddens or (2 * n_in + 1)
        self.squeeze_1d_output = squeeze_1d_output

        if isinstance(base_1d_module, nn.Module):
            def create_module(): return copy.deepcopy(base_1d_module)
        else:
            create_module = base_1d_module

        self.hiddens = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleDict({
                    'preprocess': nn.ModuleList([
                        create_module()
                        for _ in range(n_in)
                    ]),
                    'output': create_module(),
                })
                for _ in range(self.n_hiddens)
            ])
            for _ in range(n_out)
        ])

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass

        Parameters
        ----------
        x : torch.Tensor
            The input data (N, n_in)

        Returns
        -------
        torch.Tensor
            The output data (N, n_out)
        """
        x = x.to(self.get_device())
        outputs = []
        for hiddens in self.hiddens:
            # initialize
            outputs_i = []
            # apply the hidden layers to the input data
            for hiddens_i in hiddens:
                # get the preprocessors and the output module
                preprocessors = hiddens_i['preprocess']
                output = hiddens_i['output']
                # preprocess the input data
                preprocessed = sum([
                    preprocessor(x[:, i].unsqueeze(1))
                    for i, preprocessor in enumerate(preprocessors)
                ]).view(x.size(0), -1)
                # get the output
                outputs_i.append(output(preprocessed))
            # aggregate the outputs
            outputs.append(sum(outputs_i).view(x.size(0), 1))
        output = torch.cat(outputs, dim=1)
        if self.n_out == 1 and self.squeeze_1d_output:
            return output.squeeze(1)
        return output
