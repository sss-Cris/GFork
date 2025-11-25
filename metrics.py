import torch

def softmax_cross_entropy(loss_fn, preds, labels):
    """
    Compute softmax cross-entropy loss for standard predictions.

    Args:
        loss_fn: A PyTorch loss function, e.g., nn.CrossEntropyLoss().
        preds (torch.Tensor): Predicted logits of shape [batch_size, num_classes].
        labels (torch.Tensor): One-hot encoded labels of shape [batch_size, num_classes].

    Returns:
        torch.Tensor: Scalar loss.
    """
    # Convert one-hot labels to class indices
    target_indices = torch.max(labels, 1)[1]
    loss = loss_fn(preds, target_indices)
    return loss

def softmax_cross_entropy_HG(loss_fn, preds, labels):
    """
    Compute softmax cross-entropy loss for hierarchical or transposed predictions.

    Args:
        loss_fn: A PyTorch loss function.
        preds (torch.Tensor): Predicted logits of shape [num_classes, batch_size].
        labels (torch.Tensor): One-hot encoded labels of shape [num_classes, batch_size].

    Returns:
        torch.Tensor: Scalar loss.
    """
    target_indices = torch.max(labels, 0)[1]
    loss = loss_fn(preds, target_indices)
    return loss

def accuracy(preds, labels):
    """
    Compute accuracy for standard predictions.

    Args:
        preds (torch.Tensor): Predicted logits [batch_size, num_classes].
        labels (torch.Tensor): One-hot encoded labels [batch_size, num_classes].

    Returns:
        float: Accuracy score between 0 and 1.
    """
    correct = torch.sum(torch.argmax(preds, 1) == torch.argmax(labels, 1))
    return correct / len(preds)

def accuracy_HG(preds, labels):
    """
    Compute accuracy for hierarchical or transposed predictions.

    Args:
        preds (torch.Tensor): Predicted logits [num_classes, batch_size].
        labels (torch.Tensor): One-hot encoded labels [num_classes, batch_size].

    Returns:
        float: Accuracy score between 0 and 1.
    """
    correct = torch.sum(torch.argmax(preds, 0) == torch.argmax(labels, 0))
    return correct / len(preds)
