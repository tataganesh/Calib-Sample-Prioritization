from abc import ABCMeta, abstractmethod
from statistics import mode
import torch
import numpy as np


def calc_entropy(probs, num_classes=10):
    e_term = 0.000001  # To avoid log(0) problems
    log_prob = torch.log(probs + e_term)
    entropy = -(probs * log_prob).sum(dim=1)
    normalized_entropy = entropy/torch.log(torch.tensor(num_classes))
    return normalized_entropy

def select_top_k(indices, num_samples):
    return indices[:num_samples]

def select_uniform(indices, num_samples, entropies=None):
    total_samples = indices.shape[0]
    selection_idx = np.arange(0, total_samples, int(total_samples/num_samples))[:num_samples]
    return indices[selection_idx]


selection_func_mapping  = {"top_k": select_top_k, "uniform": select_uniform}

class UncertaintySampling(metaclass=ABCMeta):
    def __init__(self, dataset_indices, select_crit='top_k', discount_factor=0, num_classes=10, use_target=False, target_entropies=None):
        print(f"Prioritizer Initialization")
        print(f"Selection Criterion: {select_crit}")
        self.discount_factor = discount_factor
        self.dataset_indices = dataset_indices  # To map from original dataset to training subset
        self.dataset_len = len(self.dataset_indices)
        self.target_entropies = target_entropies
        self.num_classes = num_classes
        self.use_target = use_target
        self.selection_function = selection_func_mapping[select_crit]

    def get_indices(self, indices_ordered, num_samples, epoch):
        return self.selection_function(indices_ordered, num_samples)


    @abstractmethod
    def get_uncertainty(self):
        pass


class LabelEntropy(UncertaintySampling):
    def get_uncertainty(self, probs):
        entropy = calc_entropy(probs, self.num_classes)
        return entropy
    
    def query(self, samples_uncertainty, num_samples, epoch):
        # Ordered indices of both current and target models stored
        if self.use_target:
            target_entropies_epoch = self.target_entropies[:, epoch]
            model_indices_ordered = torch.argsort(target_entropies_epoch[self.dataset_indices], descending=True)
        else:
            model_indices_ordered = torch.argsort(samples_uncertainty[self.dataset_indices], descending=True)
        return self.selection_function(model_indices_ordered, num_samples)


class RandomSampling(UncertaintySampling):
    def query(self, samples_uncertainty, num_samples, epoch):
        random_indices = torch.randperm(self.dataset_len)
        return random_indices[:num_samples]

    def get_uncertainty(self, probs):
        entropy = calc_entropy(probs, self.num_classes)
        return entropy


def prioritizer_factory(al_method):
    method_mapping = {'entropy': LabelEntropy, 'random': RandomSampling}
    return method_mapping[al_method]
