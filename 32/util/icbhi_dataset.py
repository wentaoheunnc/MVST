import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from copy import deepcopy
from PIL import Image

from .icbhi_util import get_annotations, generate_fbank, get_individual_cycles_torchaudio, cut_pad_sample_torchaudio
from .augmentation import augment_raw_audio


class ICBHIDataset(Dataset):
    def __init__(self, train_flag, transform, args, print_flag=True, mean_std=False):
        data_folder = os.path.join(args.data_folder, 'icbhi_dataset')
        test_fold = args.test_fold
        
        self.data_folder = data_folder
        self.train_flag = train_flag
        self.split = 'train' if train_flag else 'test'
        self.transform = transform
        self.args = args
        self.mean_std = mean_std

        # parameters for spectrograms
        self.sample_rate = args.sample_rate
        self.desired_length = args.desired_length
        self.pad_types = args.pad_types
        self.nfft = args.nfft
        self.hop = self.nfft // 2
        self.n_mels = args.n_mels
        self.f_min = 50
        self.f_max = 2000

        filenames = os.listdir(data_folder)
        filenames =set([f.strip().split('.')[0] for f in filenames if '.wav' in f or '.txt' in f])
        filenames = sorted(filenames)

        patient_dict = {}
        indices = [i for i, file in enumerate(filenames)]
        random.Random(1).shuffle(indices)
        train_size = int(len(indices) * 0.6)
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]
        train_files = [filenames[i] for i in train_idx]
        test_files = [filenames[i] for i in test_idx]
        for f in train_files:
            if train_flag:
                patient_dict[f] = 'train'
        for f in test_files:
            if not train_flag:
                patient_dict[f] = 'test'

        if print_flag:
            print('*' * 20)
            print('Train and test 60-40% split with test_fold {}'.format(test_fold))
            print('File number in {} dataset: {}'.format(self.split, len(patient_dict)))

        annotation_dict = get_annotations(args, data_folder)

        self.filenames = []
        for f in filenames:
            idx = f.split('_')[0] if test_fold in ['0', '1', '2', '3', '4'] else f
            if args.stetho_id >= 0:
                if idx in patient_dict and self.file_to_device[f] == args.stetho_id:
                    self.filenames.append(f)
            else:
                if idx in patient_dict:
                    self.filenames.append(f)
        
        self.audio_data = []
        self.labels = []

        if print_flag:
            print('*' * 20)  
            print("Extracting individual breathing cycles..")

        self.cycle_list = []
        self.filename_to_label = {}
        self.classwise_cycle_list = [[] for _ in range(args.n_cls)]
        self.filenames.sort()

        for idx, filename in enumerate(self.filenames):
            self.filename_to_label[filename] = []
            sample_data = get_individual_cycles_torchaudio(args, annotation_dict[filename], data_folder, filename, args.sample_rate, args.n_cls)
            cycles_with_labels = [(data[0], data[1]) for data in sample_data]
            self.cycle_list.extend(cycles_with_labels)
            for d in cycles_with_labels:
                # {filename: [label for cycle 1, ...]}
                self.filename_to_label[filename].append(d[1])
                self.classwise_cycle_list[d[1]].append(d)
                
        for sample in self.cycle_list:
            self.audio_data.append(sample)

        self.class_nums = np.zeros(args.n_cls)
        for sample in self.audio_data:
            self.class_nums[sample[1]] += 1
            self.labels.append(sample[1])
        self.class_ratio = self.class_nums / sum(self.class_nums) * 100
        
        if print_flag:
            print('[Preprocessed {} dataset information]'.format(self.split))
            print('total number of audio data: {}'.format(len(self.audio_data)))
            for i, (n, p) in enumerate(zip(self.class_nums, self.class_ratio)):
                print('Class {} {:<9}: {:<4} ({:.1f}%)'.format(i, '('+args.cls_list[i]+')', int(n), p))    
        
        self.audio_images = []
        for index in range(len(self.audio_data)):
            audio, label = self.audio_data[index][0], self.audio_data[index][1]

            audio_image = []
            for aug_idx in range(self.args.raw_augment+1): 
                if aug_idx > 0:
                    if self.train_flag and not mean_std:
                        audio = augment_raw_audio(audio, self.sample_rate, self.args)
                        audio = cut_pad_sample_torchaudio(torch.tensor(audio), args)
                    else:
                        audio_image.append(None)
                        continue
                
                image = generate_fbank(audio, self.sample_rate, n_mels=self.n_mels)
                audio_image.append(image)
            self.audio_images.append((audio_image, label))

        self.h, self.w, _ = self.audio_images[0][0][0].shape

    def __getitem__(self, index):
        audio_images, label = self.audio_images[index][0], self.audio_images[index][1]

        if self.args.raw_augment and self.train_flag and not self.mean_std:
            aug_idx = random.randint(0, self.args.raw_augment)
            audio_image = audio_images[aug_idx]
        else:
            audio_image = audio_images[0]
        
        if self.transform is not None:
            audio_image = self.transform(audio_image)
        
        return audio_image, label

    def __len__(self):
        return len(self.audio_data)