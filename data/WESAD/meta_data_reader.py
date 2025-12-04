import numpy as np
from utils.filter import butter_bandpass_filter, detrend


def shuffle_data(ppgs, edas, labels):
    num = len(labels)
    shuffle_index = np.random.permutation(np.arange(num))
    shuffled_ppgs = ppgs[shuffle_index]
    shuffled_edas = edas[shuffle_index]
    shuffled_labels = labels[shuffle_index]
    return shuffled_ppgs, shuffled_edas, shuffled_labels


class BatchSignalGenerator:
    def __init__(self, args, stage, data, person):
        if stage not in ['train', 'val', 'test']:
            assert ValueError('invalid stage!')
        self.configuration(args, stage, person)
        self.args = args
        self.data = data
        self.binary = True if args.y_dim == 2 else False
        self.task = args.ubfc_phys_task
        self.dataset_name = args.dataset_name
        self.load_data()

    def configuration(self, args, stage, person):
        self.batch_size = args.batch_size
        self.current_index = -1
        self.person = person
        self.stage = stage
        self.shuffled = False

    def z(self, signal):
        mean = np.average(signal)
        signal = signal - mean
        std = np.std(signal)
        signal = signal / std
        return signal

    def normalize(self, ppgs, edas):
        ppgs_norm = []
        edas_norm = []
        for ppg, eda in zip(ppgs, edas):
            ppgs_norm.append(self.z(ppg))
            edas_norm.append(self.z(eda))
        return np.stack(ppgs_norm), np.stack(edas_norm)

    def load_data(self):
        needed_data = None

        if self.dataset_name == "ubfc_phys":
            if self.binary:
                w_label = [1, self.task]
            else:
                w_label = [1, 2, 3]
        elif self.dataset_name == "wesad":
            w_label = [0, 1, 2]

        for label in w_label:
            selected_data = self.data[(self.data[:, 0] == self.person) & (self.data[:, 1] == label)]
            if needed_data is None:
                needed_data = selected_data
            else:
                needed_data = np.concatenate((needed_data, selected_data))
        ppgs = np.array([ppg.astype(np.float32) for ppg in needed_data[:, 2]])
        edas = np.array([ppg.astype(np.float32) for ppg in needed_data[:, 5]])
        self.ppgs, self.edas = self.normalize(ppgs, edas)

        self.ppgs = self.ppgs[:, np.newaxis, :]
        self.edas = self.edas[:, np.newaxis, :]

        if self.dataset_name == "ubfc_phys":
            if self.binary:
                self.labels = np.array([(0 if l == 1 else 1) for l in needed_data[:, 1]]).astype(np.uint8)
            else:
                self.labels = np.array([l - 1 for l in needed_data[:, 1]]).astype(np.uint8)
        elif self.dataset_name == "wesad":
            if self.binary:
                self.labels = np.array([(l if l == 1 else 0) for l in needed_data[:, 1]]).astype(np.uint8)
            else:
                self.labels = np.array([l for l in needed_data[:, 1]]).astype(np.uint8)

        self.file_num_train = len(self.labels)
        print('data num loaded:', self.file_num_train)
        if self.stage is 'train':
            self.ppgs, self.edas, self.labels = shuffle_data(self.ppgs, self.edas, self.labels)

    def get_signals_labels_batch(self):
        ppgs = []
        edas = []
        labels = []
        for index in range(self.batch_size):
            self.current_index += 1

            # void over flow
            if self.current_index > self.file_num_train - 1:
                self.current_index %= self.file_num_train

                self.ppgs, self.edas, self.labels = shuffle_data(self.ppgs, self.edas, self.labels)

            ppgs.append(self.ppgs[self.current_index])
            edas.append(self.edas[self.current_index])
            labels.append(self.labels[self.current_index])

        ppgs = np.stack(ppgs)
        edas = np.stack(edas)
        labels = np.stack(labels)

        return ppgs, edas, labels
