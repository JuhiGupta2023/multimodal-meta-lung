# Placeholder metrics utility
def compute_accuracy(preds, labels):
    return (preds == labels).sum() / len(labels)
