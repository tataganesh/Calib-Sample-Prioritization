from numpy.core.fromnumeric import shape
import torch
from sklearn.metrics import roc_auc_score


def calibration(y, p_mean, num_bins=10):

    # Compute for every test sample x, the predicted class.
    class_pred = torch.argmax(p_mean, dim=1).cpu()
    # and the confidence (probability) associated with it.
    conf, _ = torch.max(p_mean, dim=1)
    # Convert y from one-hot encoding to the number of the class
    y = torch.argmax(y, dim=1)
    acc_tab = torch.zeros(num_bins)  # empirical (true) confidence
    mean_conf = torch.zeros(num_bins)  # predicted confidence
    nb_items_bin = torch.zeros(num_bins)  # number of items in the bins
    tau_tab = torch.linspace(0, 1, num_bins+1)  # confidence bins
    for i in torch.arange(num_bins):  # iterate over the bins
        # Select the items where the predicted max probability falls in the bin [tau_tab[i], tau_tab[i + 1)]
        sec = (tau_tab[i+1] > conf) & (conf >= tau_tab[i])
        nb_items_bin[i] = torch.sum(sec)  # Number of items in the bin
        # Select the predicted classes, and the true classes
        class_pred_sec, y_sec = class_pred[sec], y[sec]
        # Average of the predicted max probabilities
        mean_conf[i] = torch.mean(conf[sec]) if nb_items_bin[i] > 0 else 0
        # Compute the empirical confidence
        acc_tab[i] = torch.mean((class_pred_sec == y_sec).float()) if nb_items_bin[i] > 0 else 0

    # Expected Calibration Error
    ece = torch.sum(torch.absolute(acc_tab - mean_conf) * nb_items_bin.float() / torch.sum(nb_items_bin))
    # Maximum Calibration Error
    mce = torch.max(torch.absolute(acc_tab - mean_conf))
    # Overconfidence Error
    oe = torch.sum(mean_conf * torch.max((mean_conf - acc_tab), torch.zeros(mean_conf.shape[0])) * nb_items_bin.float() / torch.sum(nb_items_bin))
    # AUROC
    auroc = 0.0
    # Reliability diagram
    rel_diag = (mean_conf, acc_tab)

    cal = {'rel_diag': rel_diag, 'bin_items': nb_items_bin, 'ece': ece, 'mce': mce, 'oe': oe, 'auroc': auroc}

    return cal
