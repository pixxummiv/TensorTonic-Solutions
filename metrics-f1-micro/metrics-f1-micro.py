def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    tp, fp, fn = 0, 0, 0
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            tp = tp + 1
        else:
            fp = fp + 1
            fn = fn + 1
    micro_f1 = (2*tp) / (2*tp+fp+fn)
    return micro_f1
        