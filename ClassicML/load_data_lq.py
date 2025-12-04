import os
import math
import numpy as np
import pandas as pd
import yaml

from data.WESAD.Uniform_distribution_person_multi import train_subject as train_subject_WESAD, \
    valid_subject as valid_subject_WESAD, test_subject as test_subject_WESAD
from data.UBFC_Phys.Uniform_distribution_person_multi import train_subject as train_subject_UBFC_Phys, \
    valid_subject as valid_subject_UBFC_Phys, test_subject as test_subject_UBFC_Phys
from data.CAN_STRESS.Uniform_distribution_person_multi_0_4 import train_subject as train_subject_CAN_STRESS, \
    valid_subject as valid_subject_CAN_STRESS, test_subject as test_subject_CAN_STRESS
from data.VerBIO.Uniform_distribution_person_multi import train_subject as train_subject_VerBIO, \
    valid_subject as valid_subject_VerBIO, test_subject as test_subject_VerBIO


class DataLoder():
    def __init__(self, dataset_name, task=None):
        self.dataset_name = dataset_name
        if dataset_name == 'wesad':
            self.train_subject, self.valid_subject, self.test_subject = train_subject_WESAD, valid_subject_WESAD, test_subject_WESAD
            # self.data = np.load('comparation/ClassicML/WESAD_CML_clip30s.npy', allow_pickle=True)
            self.data = np.load('WESAD_CLM_clip30s_lowq.npy', allow_pickle=True)
            self.k = 5
        elif dataset_name == 'ubfc_phys':
            self.train_subject, self.valid_subject, self.test_subject = train_subject_UBFC_Phys, valid_subject_UBFC_Phys, test_subject_UBFC_Phys
            # self.data = np.load('comparation/ClassicML/UBFC_Phys_CLM_clip30s.npy', allow_pickle=True)
            self.data = np.load('UBFC_Phys_CLM_clip30s_lowq.npy', allow_pickle=True)
            self.task = task
            self.k = 7
        # elif dataset_name == 'can_stress':
        #     self.train_subject, self.valid_subject, self.test_subject = train_subject_CAN_STRESS, valid_subject_CAN_STRESS, test_subject_CAN_STRESS
        #     self.data = np.load('CAN_STRESS_CLM_clip30s.npy', allow_pickle=True)
        #     # with open("CAN_STRESS_CLM_clip30s.pkl", "rb") as f:
        #     #     data = pickle.load(f)
        #     self.k = 5
        elif dataset_name == 'verbio':
            self.train_subject, self.valid_subject, self.test_subject = train_subject_VerBIO, valid_subject_VerBIO, test_subject_VerBIO
            self.data = np.load('VerBIO_CLM_clip30s_lowq.npy', allow_pickle=True)
            # with open("CAN_STRESS_CLM_clip30s.pkl", "rb") as f:
            #     data = pickle.load(f)
            self.k = 6

    def load(self, phase, fold):
        if phase == 'train':
            person_list = self.train_subject[fold]
        elif phase == 'valid':
            person_list = self.valid_subject[fold]
        elif phase == 'test':
            person_list = self.test_subject[fold]
        elif phase == 'cross':
            person_list = self.train_subject[fold] + self.valid_subject[fold] + self.test_subject[fold]
        needed_data = None
        if self.dataset_name == "wesad":
            for person in person_list:
                for label in [0, 1, 2]:
                    selected_data = self.data[(self.data[:, 0] == person) & (self.data[:, 1] == label)]
                    if needed_data is None:
                        needed_data = selected_data
                    else:
                        needed_data = np.concatenate((needed_data, selected_data))
            labels = np.array([(l if l == 1 else 0) for l in needed_data[:, 1]]).astype(np.uint8)
        elif self.dataset_name == "ubfc_phys":
            w_label = [1, 2, 3]
            for person in person_list:
                for label in w_label:
                    selected_data = self.data[(self.data[:, 0] == person) & (self.data[:, 1] == label)]
                    if needed_data is None:
                        needed_data = selected_data
                    else:
                        needed_data = np.concatenate((needed_data, selected_data))
            labels = np.array([(0 if l == 1 else 1) for l in needed_data[:, 1]]).astype(np.uint8)
        # elif self.dataset_name == "can_stress":
        #     for person in person_list:
        #         for label in [0, 1]:
        #             selected_data = self.data[(self.data[:, 0] == person) & (self.data[:, 1] == label)]
        #             if needed_data is None:
        #                 needed_data = selected_data
        #             else:
        #                 needed_data = np.concatenate((needed_data, selected_data))
        #     labels = np.array([l for l in needed_data[:, 1]]).astype(np.uint8)
        elif self.dataset_name == "verbio":
            for person in person_list:
                for label in [0, 1]:
                    selected_data = self.data[(self.data[:, 0] == person) & (self.data[:, 1] == label)]
                    if needed_data is None:
                        needed_data = selected_data
                    else:
                        needed_data = np.concatenate((needed_data, selected_data))
            labels = np.array([l for l in needed_data[:, 1]]).astype(np.uint8)
        persons = np.array(needed_data[:, 0])

        features = np.stack([np.array(f).astype(np.float32) for f in needed_data[:, 2]])
        return persons, features, labels
