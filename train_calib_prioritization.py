import os
import sys
import argparse
from pprint import pprint
import copy
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Models'))
from regularizers.mixup import mixup_criterion, mixup_data
from calibration_metrics import calibration
from plot_utils import plot_reliability_diagram
from datasets import get_dataset
from Models.models import get_model
from sampling_methods import prioritizer_factory
import DATASET_CONSTANTS
from focal_loss import FocalLoss
import json


labels_mapping = {
    "cifar10": DATASET_CONSTANTS.CIFAR10_LABEL_MAPPING,
    "cifar100": DATASET_CONSTANTS.CIFAR100_LABEL_MAPPING,
    }

CALIB_METRICS = ['ece', 'mce', 'oe', 'auroc']


def print_model_config(model, optimizer):
    print('===========================')
    print(f'model class: {model.__class__}')
    for m in model.children():
        print(m)
    print('---------------------------')
    print(f'optim class: {optimizer.__class__}')
    print('===========================')


class TrainNetwork:
    def __init__(self, args):
        self.random_state = args.random_state
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        self.data_path = args.data_path
        self.dataset = args.dataset
        self.num_samples = args.num_samples
        self.batch_size = args.batch_size
        self.warmup_epochs = args.warmup_epochs
        self.epochs = args.epochs
        self.num_subset = args.num_subset
        self.num_bins = args.num_bins
        self.select_crit = args.select_crit
        self.num_train_percent = args.num_train_perc
        self.top_k_important_samples = torch.randperm(self.num_subset)  # Initial indices of top k important samples
        self.important_samples_per_epoch = list()
        self.dataset_labels_mapping = labels_mapping[self.dataset]
        self.num_classes = len(self.dataset_labels_mapping)
        self.imp_labels_epoch = {label_name: [] for key, label_name in self.dataset_labels_mapping.items()}
        self.mixup = False
        self.mixup_alpha = args.mixup_alpha
        self.fl_gamma = args.fl_gamma
        self.use_target = args.use_target
        self.target_ents_path = args.target_ents_path
        self.loss_function = nn.CrossEntropyLoss()
        if args.calibration == 'label_smoothing':
            self.loss_function = nn.CrossEntropyLoss(label_smoothing=args.ls_factor)
        elif args.calibration == 'mixup':
            self.mixup = True
        elif args.calibration == "focal_loss":
            self.loss_function = FocalLoss(gamma=self.fl_gamma)


        self.target_entropies = None
        # target info
        if self.use_target:
            target_entropies_dict = json.load(open(self.target_ents_path))
            self.target_entropies = torch.tensor(target_entropies_dict["data"])

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Device - {self.device}')

        # Construct Dataset with transforms
        self.full_dataset, self.train_set, self.val_set, self.test_set = get_dataset(self.dataset, self.data_path, self.num_samples, self.num_train_percent)

        self.num_train_samples = len(self.train_set)
        self.num_val_samples = len(self.val_set)
        self.num_test_samples = len(self.test_set)
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

        print(f'Number of training samples - {self.num_train_samples}')
        print(f'Number of validation samples - {self.num_val_samples }')
        print(f'Number of test samples - {self.num_test_samples}')

        # Train-Val-Test Distribution
        get_labels = lambda dataset: [label for _, label, _ in dataset]
        train_targets = get_labels(self.train_set)
        val_targets = get_labels(self.val_set)
        test_targets = get_labels(self.test_set)
        print(f'Train distribution \n {pd.value_counts(train_targets)}')
        print(f'Validation distribution \n {pd.value_counts(val_targets)}')
        print(f'Test distribution \n {pd.value_counts(test_targets)}')

        if not self.num_samples:
            self.num_samples = self.num_train_samples + self.num_val_samples

        # Tracking importance scores
        self.sample_importance = torch.full((self.num_samples,), fill_value=-1, dtype=torch.float32)
        self.sample_importance[self.train_set.indices] = 0
        # Track samples selected in each epoch
        self.selected_samples = torch.zeros((self.num_samples, self.epochs), dtype=torch.float32)
        self.sample_importance_all_epochs = torch.zeros((self.num_samples, self.epochs), dtype=torch.float32)
        method = prioritizer_factory(args.importance_criterion)
        self.priotizer = method(self.train_set.indices, self.select_crit, num_classes=self.num_classes, use_target=self.use_target, target_entropies=self.target_entropies)
        self.scheduler_type = args.scheduler_type

        # Configure model
        self.model = get_model(args.nn_arch, num_classes=self.num_classes).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        if self.scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
            print(self.scheduler)
        elif self.scheduler_type == 'multistep':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 120, 160], gamma=0.2)

        print_model_config(self.model, self.optimizer)
        self.softmax = torch.nn.Softmax(dim=1)


    def get_accuracy(self, loader, num_samples):
        num_correct = 0.0
        total_loss = 0.0
        pred_probs, y = list(), list()
        for inputs, labels, sample_indices in loader:
            inputs, labels, sample_indices = inputs.to(self.device), labels.to(self.device), sample_indices.to(self.device)
            output = self.model(inputs)
            pred_probs.append(self.softmax(output))
            train_predictions = output.argmax(dim=1)
            loss = self.loss_function(output, labels)
            total_loss += loss.item()
            num_correct += (labels == train_predictions).sum()
            y.extend(labels)
        accuracy = num_correct / num_samples * 100.0
        pred_probs = torch.cat(pred_probs)
        y_onehot = F.one_hot(torch.tensor(y))
        avg_loss = total_loss / (num_samples // self.batch_size)
        return accuracy, pred_probs, y_onehot, avg_loss

    def train(self):
        # Train
        train_loss = 0.0
        calibration_metrics_train, calibration_metrics_val = dict(), dict()
        best_model = None
        lowest_val_loss_epoch = None
        lowest_val_loss = 10000
        train_loader_orig = self.train_loader
        for epoch in range(self.epochs):
            self.model.train()
            all_metrics = dict()
            common_samples_percent = 0

            # Check if sample priortization is to be done
            if epoch >= self.warmup_epochs and self.num_subset:
                self.top_k_important_samples = self.priotizer.query(self.sample_importance, self.num_subset, epoch)
                num_top_k = len(self.top_k_important_samples)
                sampler = SubsetRandomSampler(self.top_k_important_samples)  # Mapping full dataset indices to training subset
                self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, sampler=sampler, shuffle=False)
                if len(self.important_samples_per_epoch):
                    common_samples_percent = np.intersect1d(self.top_k_important_samples, self.important_samples_per_epoch[-1]).shape[0] / float(self.num_subset) * 100.0
                    print(f'Common samples between consecutive epochs - {common_samples_percent}')
                self.important_samples_per_epoch.append(self.top_k_important_samples)
                # Determine class frequency in selected samples
                important_samples_labels = torch.tensor([self.train_set[ind.item()][1] for ind in self.top_k_important_samples])  # Find labels of imp samples
                unique_labels, counts = torch.unique(important_samples_labels, return_counts=True)  # Count occurences of labels
                # Convert counts to percentages
                class_frequencies = {self.dataset_labels_mapping[ind.item()]: counts[i]*100.0/num_top_k for i, ind in enumerate(unique_labels)}

                for label_ID, label_name in self.dataset_labels_mapping.items():
                    if f'{label_name}' not in class_frequencies:
                        class_frequencies[label_name] = torch.tensor(0)
                    self.imp_labels_epoch[label_name].append(class_frequencies[label_name])

                class_frequencies = {f"class_frequency/{label_name}":freq_perc for label_name, freq_perc in class_frequencies.items()}
                all_metrics.update(class_frequencies)
                all_metrics['train/intersection_between_epochs'] = common_samples_percent

                original_dataset_indices = [self.train_set.indices[index] for index in self.top_k_important_samples]
                self.selected_samples[original_dataset_indices, epoch] = 1

            print(f'Train samples - {len(self.train_loader.sampler)}')
            tk0 = tqdm(self.train_loader)
            tk0.set_description(f'Train Epoch {epoch}')
            num_correct = 0
            train_loss_full = 0.0
            for inputs, labels, sample_indices in tk0:
                inputs, labels, sample_indices = inputs.to(self.device), labels.to(self.device), sample_indices.to(self.device)
                self.optimizer.zero_grad()
                if self.mixup:
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=self.mixup_alpha)
                
                train_output = self.model(inputs)
                if self.mixup:
                    train_loss = mixup_criterion(self.loss_function, train_output, targets_a, targets_b, lam)
                else:
                    train_loss = self.loss_function(train_output, labels)

                train_loss_full += train_loss.item()
                train_loss.backward()
                self.optimizer.step()
                tk0.set_postfix(loss=train_loss.item())
                self.sample_importance[sample_indices] = self.priotizer.get_uncertainty(self.softmax(train_output).detach().cpu())
            if self.scheduler_type:
                print('Scheduler step invoked')
                self.scheduler.step()

            train_loss = train_loss_full / (len(self.train_loader.dataset) // self.batch_size)
            print(f'Train Loss: {train_loss:.6f}', end='\t')

            self.sample_importance_all_epochs[:, epoch] = self.sample_importance
            self.model.eval()
            with torch.no_grad():
                # Train Accuracy
                train_accuracy, train_pred_probs, y_train_onehot, _ = self.get_accuracy(train_loader_orig, self.num_train_samples )
                print(f'Train Accuracy: {train_accuracy:.6f}')

                # Calibration during train
                calibration_metrics_train = calibration(y_train_onehot, train_pred_probs.cpu(), self.num_bins)

                # Validation
                val_accuracy, val_pred_probs, y_val_onehot, val_loss = self.get_accuracy(self.val_loader, self.num_val_samples)
                print(f'Validation Loss: {val_loss:.6f} \t Validation Accuracy: {val_accuracy:.6f}\n')

                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    lowest_val_loss_epoch = epoch
                    best_model = copy.deepcopy(self.model)

                # Calibration during validation
                calibration_metrics_val = calibration(y_val_onehot, val_pred_probs.cpu(), self.num_bins)

            for metric in CALIB_METRICS:
                all_metrics["train/" + metric] = calibration_metrics_train[metric]
                all_metrics["val/" + metric] = calibration_metrics_val[metric]

            all_metrics['train/loss'] = train_loss
            all_metrics['train/acc'] = train_accuracy
            all_metrics['val/loss'] = val_loss
            all_metrics['val/acc'] = val_accuracy
        

        # Train Reliability Diagram and Bin items
        rel_diag_figpath_train = plot_reliability_diagram(calibration_metrics_train['rel_diag'],
                                                          'Train_Reliability_Diagram', n_bins=self.num_bins)

        # Validation Reliability Diagram and Bin items
        rel_diag_figpath_val = plot_reliability_diagram(calibration_metrics_val['rel_diag'],
                                                        'Val_Reliability_Diagram', n_bins=self.num_bins)

        self.model.eval()
        with torch.no_grad():
            # Test
            # Best model from validation set and last epoch
            test_loss_best_full, test_loss_last_full = 0.0, 0.0
            num_correct_best_model, num_correct_last_model = 0, 0
            test_pred_probs_best, test_pred_probs_last, y_test = list(), list(), list()
            for inputs, labels, sample_indices in self.test_loader:
                inputs, labels, sample_indices = inputs.to(self.device), labels.to(self.device), sample_indices.to(self.device)

                # Validation set best model
                test_output = best_model(inputs)
                test_pred_probs_best.append(self.softmax(test_output))
                test_predictions = test_output.argmax(dim=1)
                batch_loss = self.loss_function(test_output, labels)
                num_correct_best_model += (labels == test_predictions).sum()
                test_loss_best_full += batch_loss.item()

                # Last epoch model
                test_output = self.model(inputs)
                test_pred_probs_last.append(self.softmax(test_output))
                test_predictions = test_output.argmax(dim=1)
                batch_loss = self.loss_function(test_output, labels)
                num_correct_last_model += (labels == test_predictions).sum()
                test_loss_last_full += batch_loss.item()

                y_test.extend(labels)

            test_loss_best = test_loss_best_full / (self.num_test_samples // self.batch_size)
            test_accuracy_best = num_correct_best_model / self.num_test_samples * 100.0
            test_loss_last = test_loss_last_full / (self.num_test_samples // self.batch_size)
            test_accuracy_last = num_correct_last_model / self.num_test_samples * 100.0
            print(f'Validation Model - Best Epoch: {lowest_val_loss_epoch} \t Last Epoch: {self.epochs-1}')
            print(f'Best Test Loss: {test_loss_best:.6f} \t Best Test Accuracy: {test_accuracy_best:.6f}')
            print(f'Last Test Loss: {test_loss_last:.6f} \t Last Test Accuracy: {test_accuracy_last:.6f}')
            test_pred_probs_last = torch.cat(test_pred_probs_last)
            y_test_onehot = F.one_hot(torch.tensor(y_test))

            calibration_metrics_test_last = dict()
            calibration_metrics_test_last = calibration(y_test_onehot, test_pred_probs_last.cpu(), self.num_bins)

        print(f"Test Accuracy - {test_accuracy_last}, Test ECE - {calibration_metrics_test_last['ece']}")
        rel_diag_figpath_test_last = plot_reliability_diagram(calibration_metrics_test_last['rel_diag'],
                                                              'Test_Last_Reliability_Diagram', self.num_bins)



parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./',
                    help='Dataset download path')
parser.add_argument('--dataset', choices=["cifar10", "cifar100"], default='cifar10',
                    help='Specify dataset for training')
parser.add_argument('--num_samples', type=int,
                    help='Total number of samples')
parser.add_argument('--num_train_perc', type=float, default=0.9,
                    help='Percentage of samples used for traning. Rest of the samples will be used for validation.')
parser.add_argument('--random_state', type=int, default=2021,
                    help='Random seed for data creation')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Mini-Batch Size')
parser.add_argument('--importance_criterion', default='entropy',
                    help='Method of sample importance [entropy, random]')
parser.add_argument('--warmup_epochs', default=1000, type=int,
                    help='Number of warmup epochs')
parser.add_argument('--epochs', default=200, type=int,
                    help='Number of epochs')
parser.add_argument('--num_subset', default=0, type=int,
                    help='Number of samples in subset')
parser.add_argument('--calibration',
                    help='Type of calibration technique [label_smoothing, mixup, focal_loss]')
parser.add_argument('--num_bins', default=15, type=int,
                    help='Number of bins for calibration')
parser.add_argument('--target_model_path',
                    help='Target model folder containing the model and top k sample indices')
parser.add_argument('--nn_arch', default='resnet_18',
                    help='Specify architecture [resnet18]')
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--scheduler_type', default='cosine',
                    help='Type of scheduler [cosine, multistep]')
parser.add_argument('--ls_factor', default=0, type=float,
                    help='Label Smoothening factor')                    
parser.add_argument('--mixup_alpha', default=0.15, type=float)
parser.add_argument('--fl_gamma', default=0, type=float, 
                    help='Focal Loss gamma')
parser.add_argument('--select_crit', default="top_k", 
                    help='Selection Criterion when choosing subset [top_k, uniform]')

parser.add_argument('--use_target', help='Use target for selecting samples', action="store_true")
parser.add_argument('--target_ents_path', help='Pass path to target entropies')
parser.add_argument('--target_arch', help='Specify target network architecture')



args = parser.parse_args()
pprint(vars(args))
if args.use_target:
    if not args.target_ents_path:
        print("Target Entropies path not specified. Exiting...")
        sys.exit()
config = vars(args)

network = TrainNetwork(args)

network.train()
