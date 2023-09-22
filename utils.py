import torch

def one_hot_encode(label, num_classes):
    """
    Perform one-hot encoding for a given label using PyTorch.

    Args:
        label (int or str): The label to be one-hot encoded.
        num_classes (int): The total number of classes.

    Returns:
        torch.Tensor: The one-hot encoding of the label.
    """
    if isinstance(label, int):
        # If the label is an integer (for numbers - MNIST numbers)
        if label < 0 or label >= num_classes:
            raise ValueError("Label is out of range for the given number of classes.")
        one_hot_encoding = torch.eye(num_classes)[label]
        return one_hot_encoding
    elif isinstance(label, str):
        # If the label is a string (for CIFAR-10)
        # Define CIFAR-10 class labels
        cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        if label not in cifar10_classes:
            raise ValueError("Label not found in CIFAR-10 class labels.")
        class_index = cifar10_classes.index(label)
        return one_hot_encode(class_index, num_classes)
    else:
        raise ValueError("Label must be an integer or string.")