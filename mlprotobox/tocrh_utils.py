from torch import nn


def freeze_module_parameters(
    module: nn.Module,
) -> nn.Module:
    """
    Freeze the parameters of a module.

    Parameters
    ----------
    module : nn.Module
        The module whose parameters to freeze.

    Returns
    -------
    nn.Module
        The module with frozen parameters.

    Warnings
    --------
    This function destructively modifies the input module.
    """
    for param in module.parameters():
        param.requires_grad = False
    return module


class DeviceCheckMixin:

    def get_device(self) -> str:
        """
        Get the device of the model.

        Returns
        -------
        str
            The device of the model.
        """
        return next(self.parameters()).device
