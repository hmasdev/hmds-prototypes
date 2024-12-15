import torch
from torch import nn
from .tocrh_utils import DeviceCheckMixin


class RBFKernel(nn.Module, DeviceCheckMixin):
    """
    A module to compute the RBF (Radial Basis Function) kernel.

    .. math::

        k(x_1, x_2) = \\sigma_f^2 \\exp\\left(-\\frac{1}{2}\\left(\\frac{\\|x_1 - x_2\\|}{\\ell}\\right)^2\\right)

    Parameters
    ----------
    length_scale : float, optional
        The length scale parameter of the kernel, by default 1.0

    sigma_f : float, optional
        The scale parameter of the kernel, by default 1.0

    Attributes
    ----------
    length_scale : torch.nn.Parameter
        The length scale parameter of the kernel

    sigma_f : torch.nn.Parameter
        The scale parameter of the kernel

    Methods
    -------
    forward(x1, x2)
        Compute the RBF kernel.
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        sigma_f: float = 1.0,
        diag_add: float = 0,
    ):
        super().__init__()
        self.length_scale = nn.Parameter(torch.tensor(length_scale, dtype=torch.float32))  # noqa
        self.sigma_f = nn.Parameter(torch.tensor(sigma_f, dtype=torch.float32))  # noqa
        self.diag_add = diag_add

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the RBF kernel.

        Parameters
        ----------
        x1 : torch.Tensor
            The first set of points (N, D)

        x2 : torch.Tensor
            The second set of points (M, D)

        Returns
        -------
        torch.Tensor
            The kernel matrix (N, M)
        """
        dists = torch.cdist(x1, x2, p=2)
        if dists.size(0) == dists.size(1):
            dists = dists + torch.eye(dists.size(0)).to(self.get_device()) * self.diag_add  # noqa
        return self.sigma_f ** 2 * torch.exp(-0.5 * (dists / self.length_scale) ** 2)


class GP1D(nn.Module, DeviceCheckMixin):
    """A 1D-Gaussian Process Regression (GPR) module

    .. math::

        \\left[f(x_1), \\cdots, f(x_M) \\right]^T \\sim \\mathcal{N}\\left(K_{X_{\\text{ind}}, \\ast}^TK_{X_{\\text{ind}}, X_{\\text{ind}}}^{-1}y_{\\text{ind}}, K_{\\ast,\\ast} - K_{X_{\\text{ind}},\\ast}K_{X_{\\text{ind}}, X_{\\text{ind}}}^{-1}K_{X_{\\text{ind}},\\ast}\\right)

    where :math:`K_{X_{\\text{ind}}, \\ast}` is the kernel matrix between inducing points and new data points,
    :math:`K_{X_{\\text{ind}}, X_{\\text{ind}}}` is the kernel matrix between inducing points, :math:`y_{\\text{ind}}` is the inducing points,
    and :math:`K_{\\ast,\\ast}` is the kernel matrix between new data points, that is,

    .. math::

        K_{X_{\\text{ind}}, X_{\\text{ind}}} = \\begin{bmatrix} k(x^{(1)}_{\\text{ind}}, x^{(1)}_{\\text{ind}}) & \\cdots & k(x^{(1)}_{\\text{ind}}, x^{(N)}_{\\text{ind}}) \\\\ \\vdots & \\ddots & \\vdots \\\\ k(x^{(N)}_{\\text{ind}}, x^{(1)}_{\\text{ind}}) & \\cdots & k(x^{(N)}_{\\text{ind}}, x^{(N)}_{\\text{ind}}) \\end{bmatrix};

        K_{X_{\\text{ind}}, \\ast} = \\begin{bmatrix} k(x^{(1)}_{\\text{ind}}, x^{(1)}) & \\cdots & k(x^{(1)}_{\\text{ind}}, x^{(M)}) \\\\ \\vdots & \\ddots & \\vdots \\\\ k(x^{(N)}_{\\text{ind}}, x^{(1)}) & \\cdots & k(x^{(N)}_{\\text{ind}}, x^{(M)}) \\end{bmatrix};

        K_{\\ast,\\ast} = \\begin{bmatrix} k(x^{(1)}, x^{(1)}) & \\cdots & k(x^{(1)}, x^{(M)}) \\\\ \\vdots & \\ddots & \\vdots \\\\ k(x^{(M)}, x^{(1)}) & \\cdots & k(x^{(M)}, x^{(M)}) \\end{bmatrix}.

    In this module, we can train the inducing points :math:`x_{\\text{ind}}`, :math:`y_{\\text{ind}}` and the kernel hyperparameters through backpropagation.

    Parameters
    ----------
    x_ind : torch.Tensor
        inducing points (N, 1)

    y_ind : torch.Tensor or None, optional
        inducing points (N, 1)
        If None, set to randomly generated samples, by default None

    kernel : torch.nn.Module or None, optional
        A module that computes the kernel between two sets of points
        If None, use the RBF kernel, by default None

    noise_std : float, optional
        Standard deviation of the observation noise, by default 1e-2

    exact_sampling : bool, optional
        If True, generate samples considering the covariance, by default False

    train_x_ind : bool, optional
        If True, make inducing x trainable, by default False

    train_y_ind : bool, optional
        If True, make inducing y trainable, by default True

    Methods
    -------
    forward(x)
        Make predictions at test points
    """  # noqa

    def __init__(
        self,
        x_ind: torch.Tensor,
        y_ind: torch.Tensor | None = None,
        kernel: nn.Module | None = None,
        noise_std=1e-2,
        exact_sampling=False,
        train_x_ind=False,
        train_y_ind=True,
    ):
        super().__init__()
        # Option to make x trainable
        self.kernel = kernel or RBFKernel()
        self.noise_std = noise_std
        self.exact_sampling = exact_sampling  # Choose sampling method

        # Initialize inducing points
        self.x_ind = nn.Parameter(x_ind.clone(), requires_grad=train_x_ind)
        if y_ind is None:
            cov = self.kernel(x_ind, x_ind)
            cov = self._add_diagonal(cov)
            cov = self._stabilize_cov(cov)
            self.y_ind = torch.distributions.MultivariateNormal(
                loc=torch.zeros_like(x_ind).view(-1),
                covariance_matrix=cov,
            ).rsample().view(-1, 1)
            self.y_ind = nn.Parameter(self.y_ind, requires_grad=train_y_ind)
        else:
            self.y_ind = nn.Parameter(y_ind.clone(), requires_grad=train_y_ind)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Make predictions at test points.

        Parameters
        ----------
        x : torch.Tensor
            new data points (M, 1)

        Returns
        -------
        torch.Tensor
            Predictions (random sample in training mode, mean in evaluation mode)
        """

        x = x.to(self.get_device())
        x_ind = self.x_ind.to(self.get_device())

        K = self.kernel(x_ind, x_ind)  # Kernel between inducing data
        # Kernel between inducing and new data
        K_s = self.kernel(x_ind, x)
        K_ss = self.kernel(x, x)  # Kernel between new data

        # Add observation noise
        K = self._add_diagonal(K)
        K_ss = self._add_diagonal(K_ss)

        # Gaussian process calculations
        K_inv = torch.linalg.inv(K)
        mu = K_s.T @ K_inv @ self.y_ind
        cov = K_ss - K_s.T @ K_inv @ K_s
        cov = self._add_diagonal(cov)
        cov = self._stabilize_cov(cov)

        if self.training:
            if self.exact_sampling:
                # Accuracy-focused: generate samples considering the covariance
                distribution = torch.distributions.MultivariateNormal(
                    loc=mu.view(-1),
                    covariance_matrix=cov,
                )
                random_sample = distribution.rsample().view(-1, 1)
            else:
                # Efficiency-focused: generate independent samples with variances for each point
                variance = torch.diag(cov).view(-1, 1)
                std_dev = variance.sqrt()
                random_sample = mu + std_dev * torch.randn_like(mu).to(self.get_device())  # noqa
            return random_sample.view(-1)
        else:
            # Evaluation mode: return the mean
            return mu.view(-1)

    @staticmethod
    def _stabilize_cov(cov: torch.Tensor) -> torch.Tensor:
        # NOTE: Ensure symmetry to stabilize numerical calculations
        return 0.5 * (cov + cov.T)

    def _add_diagonal(self, K: torch.Tensor) -> torch.Tensor:
        return K + torch.eye(K.size(0), device=K.device) * self.noise_std ** 2
