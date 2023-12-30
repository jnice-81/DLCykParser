import torch
from torch.utils.data import Dataset
import json
import matplotlib.pyplot as plt
import numpy as np

char_to_index = {'a': 1, 'b': 2}

class CFGDataset(Dataset):
    def __init__(self, dataset_path, data_type, padding=True):
        with open(dataset_path, "r") as file:
            dataset = json.load(file)
        # self.positive_samples = dataset["pos"]
        self.positive_samples = [[char_to_index[char] for char in sequence] for sequence in dataset[data_type]["pos"]]
        # self.negative_samples = dataset["neg"]
        self.negative_samples = [[char_to_index[char] for char in sequence] for sequence in dataset[data_type]["neg"]]

        # Pad sequences to the same length
        if padding:
            #max_length = max([len(sequence) for sequence in self.positive_samples + self.negative_samples])
            max_length = 100
            for sequence in self.positive_samples:
                sequence += [0] * (max_length - len(sequence))
            for sequence in self.negative_samples:
                sequence += [0] * (max_length - len(sequence))
        self.seq = self.positive_samples + self.negative_samples

        self.labels = torch.cat([torch.ones(len(self.positive_samples)), torch.zeros(len(self.negative_samples))]).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # sample = self.seq_tensor[idx], self.labels[idx]
        return torch.tensor(self.seq[idx]), self.labels[idx]
    


# if run this file directly, it will plot the histogram of the text length distribution
def main():
    path = "test_dataset.json"
    dataset = CFGDataset(path, padding=False)

    text_lengths = np.array([len(sample[0]) for sample in dataset])

    # Calculate mean and standard deviation
    mean_length = np.mean(text_lengths)
    std_dev = np.std(text_lengths)

    # Define the interval (mean ± 1 standard deviation)
    lower_bound = mean_length - 2*std_dev
    upper_bound = mean_length + 2*std_dev

    # Plotting the histogram with mean and interval lines
    plt.hist(text_lengths, bins=30, color='blue', edgecolor='black')
    plt.axvline(mean_length, color='red', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(lower_bound, color='green', linestyle='dashed', linewidth=2, label='Lower Bound (Mean - 2 SD)')
    plt.axvline(upper_bound, color='green', linestyle='dashed', linewidth=2, label='Upper Bound (Mean + 2 SD)')

    plt.title('Text Length Distribution with Mean and 2 SD Interval')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

