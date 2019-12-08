import pdb


def compute_accuracy(pred, gt):
    assert len(pred) == len(gt)
    correct_count = 0
    for i in range(len(gt)):
        if pred[i] == gt[i]:
            correct_count += 1
    return correct_count / len(gt)
