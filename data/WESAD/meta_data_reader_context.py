import numpy as np
from utils.filter import butter_bandpass_filter, detrend


def shuffle_data(samples, labels):
    num = len(labels)
    shuffle_index = np.random.permutation(np.arange(num))
    shuffled_samples = samples[shuffle_index]
    shuffled_labels = labels[shuffle_index]
    return shuffled_samples, shuffled_labels


class BatchSignalGenerator:
    def __init__(self, args, stage, binary, data, person, context_num):
        if stage not in ['train', 'val', 'test']:
            assert ValueError('invalid stage!')
        self.configuration(args, stage, person)
        self.args = args
        self.data = data
        self.binary = binary
        self.context_num = context_num
        self.load_data()

    def configuration(self, args, stage, person):
        self.batch_size = args.batch_size
        self.current_index = -1
        self.person = person
        self.stage = stage
        self.shuffled = False

    def normalize(self, signals):
        signals_norm = []
        for signal in signals:
            mean = np.average(signal)
            signal = signal - mean
            std = np.std(signal)
            signal = signal / std
            signal = butter_bandpass_filter(signal, self.args.fps, self.args.cutoff_low, self.args.cutoff_high,
                                            self.args.bwf_order)
            signals_norm.append(signal)
        return np.stack(signals_norm)

    def load_data(self):
        needed_data = None
        for label in [0, 1, 2]:
            selected_data = self.data[(self.data[:, 0] == self.person) & (self.data[:, 1] == label)]
            if needed_data is None:
                needed_data = selected_data
            else:
                needed_data = np.concatenate((needed_data, selected_data))
        self.signals = self.normalize(np.array([ppg.astype(np.float32) for ppg in needed_data[:, 2]]))
        self.signals = self.signals[:, np.newaxis, :]

        if self.binary:
            self.labels = np.array([(l if l == 1 else 0) for l in needed_data[:, 1]]).astype(np.uint8)
        else:
            self.labels = np.array([l for l in needed_data[:, 1]]).astype(np.uint8)

        self.file_num_train = len(self.labels)
        print('data num loaded:', self.file_num_train)
        if self.stage is 'train':
            self.signals, self.labels = shuffle_data(samples=self.signals, labels=self.labels)

    def get_signals_labels_batch(self):
        signals = []
        labels = []

        for index in range(self.batch_size):
            self.current_index += 1

            # void over flow
            if self.current_index > self.file_num_train - 1:
                self.current_index %= self.file_num_train

                self.signals, self.labels = shuffle_data(samples=self.signals, labels=self.labels)

            signals.append(self.signals[self.current_index])
            labels.append(self.labels[self.current_index])

        signals = np.stack(signals)
        labels = np.stack(labels)

        labels_baseline_index = np.arange(len(labels))[labels == 0]
        labels_baseline_index_choice = np.random.choice(labels_baseline_index, self.context_num, replace=False)

        signals_context = signals[labels_baseline_index_choice]
        labels_context = labels[labels_baseline_index_choice]

        signals = np.array([signals[i] for i in range(len(signals)) if i not in labels_baseline_index_choice])
        labels = np.array([labels[i] for i in range(len(labels)) if i not in labels_baseline_index_choice])

        return signals, labels, signals_context, labels_context
