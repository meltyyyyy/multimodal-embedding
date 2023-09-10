import torch


def correlation(X: torch.Tensor, Y: torch.Tensor):
    """
    Compute the correlation coefficient between two tensors along the specified axis.

    Args:
        X (torch.Tensor): The first input tensor.
        Y (torch.Tensor): The second input tensor.

    Returns:
        float: The computed correlation coefficient.

    Note:
        Both tensors `X` and `Y` should have the same shape and dimension.

    Examples:
        >>> X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> Y = torch.tensor([[1.0, 2.0], [2.0, 3.0]])
        >>> correlation(X, Y)
        0.999...

    """
    corr = torch.mean(torch.tensor([torch.corrcoef(torch.cat([x, y], axis=0))[0, 1] for x, y in zip(X, Y)])).item()
    return corr
