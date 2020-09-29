import torch


def errors(output, target, topk=(1,)):
    """
    Computes the error rate over the k-top predictions for the specified values of k
    Params:
        output (torch.tensor = None): shape (batch_size, num_classes), a batch of outputs of the model
        target (torch.tensor = None): shape (batch_size,), the labels for the batch of outputs
        topk (tuple = (1,)): top-k accuracies needed
    Returns:
        errs (list): holding torch tensors of shape (1, x) representing top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)  # int
        batch_size = target.size(0)  # int

        _, pred = output.topk(maxk, 1, True, True)  # shape (batch_size, maxk)
        pred = pred.t()  # shape (maxk, batch_size)
        correct = pred.eq(target.unsqueeze(0).expand_as(pred))  # shape (maxk, batch_size)

        errs = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)  # shape (1,)
            errs.append(1 - correct_k.mul(1. / batch_size))

    return errs


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    # test
    outputs = torch.tensor([
        [0.3, 0.1, 0.1, 0.2, 0.1, 0.2],
        [0.1, 0.2, 0.4, 0.1, 0.1, 0.1],
        [0.2, 0.3, 0.09, 0.3, 0.11, 0.2]
    ])
    targets = torch.tensor([0, 1, 2])
    errs = errors(outputs, targets, topk=(1, 5))
    print(errs)
