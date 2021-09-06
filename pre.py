import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from python_speech_features import sigproc, fbank, logfbank
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from tqdm import tqdm
import glob
import os
import pickle
import random
import time
import math
import logging
import datetime
import pandas as pd

class Cnn_Transformer(nn.Module):
    def __init__(self, num_emotions):
        super().__init__()
        self.transformer_maxpool = nn.MaxPool2d(kernel_size=[1, 3], stride=[1, 3])
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=40,
            nhead=4,
            dim_feedforward=512,
            dropout=0.5,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=6)
        self.fc1_linear = nn.Linear(40, num_emotions)
        self.softmax_out = nn.Softmax(dim=1)

    def forward(self, x):     # 输入的形状为32*1*40*63
        x_maxpool = self.transformer_maxpool(x)   # x_maxpool为32*1*40*15
        x_maxpool_reduced = torch.squeeze(x_maxpool, 1)  # x_maxpool_reduced为32*40*15
        x = x_maxpool_reduced.permute(2, 0, 1)  # x 为 15*32*40
        transformer_output = self.transformer_encoder(x) #  transformer_output 15*32*40
        transformer_embedding = torch.mean(transformer_output, dim=0)  #  transformer_embedding 32*40
        complete_embedding = torch.cat([transformer_embedding], dim=1) #  transformer_embedding 32*820
        output_logits = self.fc1_linear(complete_embedding)
        output_softmax = self.softmax_out(output_logits)
        return output_logits, output_softmax, transformer_embedding

# 设定随机种子，使每一次的随机数都是一样的
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    SEED = 0
    setup_seed(SEED)
    attention_head = 4
    attention_hidden = 32
    learning_rate = 0.001
    Epochs = 50
    BATCH_SIZE = 32
    FEATURES_TO_USE = 'mfcc'  # {'mfcc' , 'logfbank','fbank','spectrogram','melspectrogram'}
    impro_or_script = 'impro'
    featuresFileName_Ravdess = 'features_{}_Ravdess.pkl'.format(FEATURES_TO_USE)
    featuresFileName = 'features_{}_{}.pkl'.format(FEATURES_TO_USE, impro_or_script)
    featuresExist = True
    toSaveFeatures = True
    # WAV_PATH = "D:\Download\IEMOCAP/"
    WAV_PATH = "D:/ravdess/Actor_*/"
    RATE = 48000
    MODEL_NAME_1 = 'HeadFusion-{}'.format(SEED)
    MODEL_NAME_2 = 'Ravdess-{}'.format(SEED)
    MODEL_PATH = 'models/{}_{}.pth'.format(MODEL_NAME_2, FEATURES_TO_USE)

    # 数据处理
    def process_data(path, t=2, train_overlap=1, val_overlap=1.6, RATE=48000, dataset='ravdess'):
        path = path.rstrip('/')
        wav_files = glob.glob(path + '/*.wav')
        meta_dict = {}
        val_dict = {}
        IEMOCAP_LABEL = {
            '01': 'neutral',
            # '02': 'frustration',
            # '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            # '06': 'fearful',
            '07': 'happy',  # excitement->happy
            # '08': 'surprised'
        }
        RAVDESS_LABEL = {
            # '01': 'surprised',
            '02': 'neutral',
            # '03': 'calm',
            '04': 'happy',
            '05': 'sad',
            '06': 'angry',
            # '07': 'fearful',
            # '08': 'disgust'
        }

        n = len(wav_files)
        train_files = []
        valid_files = []
        train_indices = list(np.random.choice(range(n), int(n * 0.8), replace=False))
        valid_indices = list(set(range(n)) - set(train_indices))
        # for i in train_indices:
        for i in train_indices:
            train_files.append(wav_files[i])
        for i in valid_indices:
            valid_files.append(wav_files[i])

        print("constructing meta dictionary for {}...".format(path))
        # 处理训练集里面的数据
        for i, wav_file in enumerate(tqdm(train_files)):
            label = str(os.path.basename(wav_file).split('-')[2])
            if (dataset == 'iemocap'):
                if (label not in IEMOCAP_LABEL):
                    continue
                if (impro_or_script != 'all' and (impro_or_script not in wav_file)):
                    continue
                label = IEMOCAP_LABEL[label]
            elif (dataset == 'ravdess'):
                if (label not in RAVDESS_LABEL):
                    continue
                label = RAVDESS_LABEL[label]
            wav_data, _ = librosa.load(wav_file, sr=RATE)
            X1 = []
            y1 = []
            index = 0
            if (t * RATE >= len(wav_data)):
                continue

            while (index + t * RATE < len(wav_data)):
                X1.append(wav_data[int(index):int(index + t * RATE)])
                y1.append(label)
                assert t - train_overlap > 0
                index += int((t - train_overlap) * RATE)
            X1 = np.array(X1)
            meta_dict[i] = {
                'X': X1,
                'y': y1,
                'path': wav_file
            }

        print("building X, y...")
        train_X = []
        train_y = []
        for k in meta_dict:
            train_X.append(meta_dict[k]['X'])
            train_y += meta_dict[k]['y']
        train_X = np.row_stack(train_X)  # 分为每一个的合并
        train_y = np.array(train_y)
        assert len(train_X) == len(train_y), "X length and y length must match! X shape: {}, y length: {}".format(
            train_X.shape, train_y.shape)

        if (val_overlap >= t):
            val_overlap = t / 2
        for i, wav_file in enumerate(tqdm(valid_files)):
            label = str(os.path.basename(wav_file).split('-')[2])
            if (dataset == 'iemocap'):
                if (label not in IEMOCAP_LABEL):
                    continue
                if (impro_or_script != 'all' and (impro_or_script not in wav_file)):
                    continue
                label = IEMOCAP_LABEL[label]
            elif (dataset == 'ravdess'):
                if (label not in RAVDESS_LABEL):
                    continue
                label = RAVDESS_LABEL[label]
            wav_data, _ = librosa.load(wav_file, sr=RATE)
            X1 = []
            y1 = []
            index = 0
            if (t * RATE >= len(wav_data)):
                continue
            while (index + t * RATE < len(wav_data)):
                X1.append(wav_data[int(index):int(index + t * RATE)])
                y1.append(label)
                index += int((t - val_overlap) * RATE)

            X1 = np.array(X1)
            val_dict[i] = {
                'X': X1,
                'y': y1,
                'path': wav_file
            }

        return train_X, train_y, val_dict

    # 特征处理
    class FeatureExtractor(object):
        def __init__(self, rate):
            self.rate = rate

        def get_features(self, features_to_use, X):
            X_features = None
            accepted_features_to_use = ("logfbank", 'mfcc', 'fbank', 'melspectrogram', 'spectrogram', 'pase')
            if features_to_use not in accepted_features_to_use:
                raise NotImplementedError("{} not in {}!".format(features_to_use, accepted_features_to_use))
            if features_to_use in ('logfbank'):
                X_features = self.get_logfbank(X)
            if features_to_use in ('mfcc',26):
                X_features = self.get_mfcc(X)
            if features_to_use in ('fbank'):
                X_features = self.get_fbank(X)
            if features_to_use in ('melspectrogram'):
                X_features = self.get_melspectrogram(X)
            if features_to_use in ('spectrogram'):
                X_features = self.get_spectrogram(X)
            if features_to_use in ('pase'):
                X_features = self.get_Pase(X)
            return X_features

        def get_logfbank(self, X):
            def _get_logfbank(x):
                out = logfbank(signal=x, samplerate=self.rate, winlen=0.040, winstep=0.010, nfft=1024, highfreq=4000,
                               nfilt=40)
                return out

            X_features = np.apply_along_axis(_get_logfbank, 1, X)
            return X_features

        def get_mfcc(self, X, n_mfcc=40):
            def _get_mfcc(x):
                mfcc_data = librosa.feature.mfcc(x, sr=self.rate, n_mfcc=n_mfcc)
                return mfcc_data

            X_features = np.apply_along_axis(_get_mfcc, 1, X)
            return X_features

        def get_fbank(self, X):
            def _get_fbank(x):
                out, _ = fbank(signal=x, samplerate=self.rate, winlen=0.040, winstep=0.010, nfft=1024)
                return out

            X_features = np.apply_along_axis(_get_fbank, 1, X)
            return X_features

        def get_melspectrogram(self, X):
            def _get_melspectrogram(x):
                mel = librosa.feature.melspectrogram(y=x, sr=self.rate)
                mel = np.log10(mel + 1e-10)
                return mel

            X_features = np.apply_along_axis(_get_melspectrogram, 1, X)
            return X_features

        def get_spectrogram(self, X):
            def _get_spectrogram(x):
                frames = sigproc.framesig(x, 640, 160)
                out = sigproc.logpowspec(frames, NFFT=3198)
                out = out.swapaxes(0, 1)
                return out[:][:400]

            X_features = np.apply_along_axis(_get_spectrogram, 1, X)
            return X_features

        def get_Pase(self, X):
            return X


    if (featuresExist == True):
        with open(featuresFileName_Ravdess, 'rb')as f:
            features = pickle.load(f)
        train_X_features = features['train_X']
        train_y = features['train_y']
        valid_features_dict = features['val_dict']
    else:
        logging.info("creating meta dict...")
        train_X, train_y, val_dict = process_data(WAV_PATH, t=2, train_overlap=1)
        print(train_X.shape)
        print(len(val_dict))

        print("getting features")
        logging.info('getting features')
        feature_extractor = FeatureExtractor(rate=RATE)
        train_X_features = feature_extractor.get_features(FEATURES_TO_USE, train_X)
        valid_features_dict = {}
        for _, i in enumerate(val_dict):
            X1 = feature_extractor.get_features(FEATURES_TO_USE, val_dict[i]['X'])
            valid_features_dict[i] = {
                'X': X1,
                'y': val_dict[i]['y']
            }
        if (toSaveFeatures == True):
            features = {'train_X': train_X_features, 'train_y': train_y,
                        'val_dict': valid_features_dict}
            with open(featuresFileName_Ravdess, 'wb') as f:
                pickle.dump(features, f)
    # 用于分类时的标签
    dict_iemocap = {
        'neutral': torch.Tensor([0]),
        'happy': torch.Tensor([1]),
        'sad': torch.Tensor([2]),
        'angry': torch.Tensor([3]),
        'calm': torch.Tensor([4]),
        'disgust': torch.Tensor([5]),
        'fearful': torch.Tensor([6]),
        'surprised': torch.Tensor([7]),
    }
    # dict_ravdess = {
    #     'surprised': torch.Tensor([0]),
    #     'neutral': torch.Tensor([1]),
    #     'calm': torch.Tensor([2]),
    #     'happy': torch.Tensor([3]),
    #     'sad': torch.Tensor([4]),
    #     'angry': torch.Tensor([5]),
    #     'fearful': torch.Tensor([6]),
    #     'disgust': torch.Tensor([7]),
    # }
    dict_ravdess = {
        # 'surprised': torch.Tensor([0]),
        'neutral': torch.Tensor([0]),
        # 'calm': torch.Tensor([2]),
        'happy': torch.Tensor([1]),
        'sad': torch.Tensor([2]),
        'angry': torch.Tensor([3]),
        # 'fearful': torch.Tensor([6]),
        # 'disgust': torch.Tensor([7]),
    }
    # 数据读取
    class DataSet(Dataset):
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y

        def __getitem__(self, index):
            x = self.X[index]
            # x = torch.from_numpy(x).unsqueeze(0)
            x = torch.from_numpy(x)
            x = x.float()
            y = self.Y[index]
            y = dict_ravdess[y]
            y = y.long()
            return x, y

        def __len__(self):
            return len(self.X)


    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件

    log_name = 'test-result/seed-{}.log'.format(SEED)
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)

    train_data = DataSet(train_X_features, train_y)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # model = HeadFusion(attention_head, attention_hidden, 4)
    model = Cnn_Transformer(num_emotions = 4)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    maxWA = 0
    maxUA = 0
    maxACC = 0
    for epoch in range(Epochs):
        model.train()
        print_loss = 0
        for _, data in enumerate(train_loader):
            x, y = data
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            out, temp1,temp2= model(x.unsqueeze(1))
            loss = criterion(out, y.squeeze(1))
            print_loss += loss.data.item() * BATCH_SIZE
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch: {}, loss: {:.4}'.format(epoch, print_loss / len(train_X_features)))

        if (epoch > 0 and epoch % 10 == 0):
            learning_rate = learning_rate / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        # validation
        model.eval()
        UA = [0, 0, 0, 0]
        num_correct = 0
        class_total = [0, 0, 0, 0]
        matrix = np.mat(np.zeros((4, 4)), dtype=int)
        # UA = [0, 0, 0, 0, 0, 0, 0, 0]
        # num_correct = 0
        # class_total = [0, 0, 0, 0, 0, 0, 0, 0]
        # matrix = np.mat(np.zeros((8, 8)), dtype=int)
        for _, i in enumerate(valid_features_dict):
            x, y = valid_features_dict[i]['X'], valid_features_dict[i]['y']
            x = torch.from_numpy(x).float()
            y = dict_ravdess[y[0]].long()
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            if (x.size(0) == 1):
                x = torch.cat((x, x), 0)
            # out, _ = model(x.unsqueeze(1))
            _, out,temp2 = model(x.unsqueeze(1))
            # out = model(x)
            pred = torch.Tensor([0, 0, 0, 0])
            # pred = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0])
            if torch.cuda.is_available():
                pred = pred.cuda()
            for j in range(out.size(0)):
                pred += out[j]
            pred = pred / out.size(0)
            pred = torch.max(pred, 0)[1]
            if (pred == y):
                num_correct += 1
            matrix[int(y), int(pred)] += 1

        for i in range(4):
            for j in range(4):
                class_total[i] += matrix[i, j]
            UA[i] = round(matrix[i, i] / class_total[i], 3)
        WA = num_correct / len(valid_features_dict)
        if (maxWA < WA):
            maxWA = WA
        if (maxUA < sum(UA) / 8):
            maxUA = sum(UA) / 8
        if (maxACC < (WA + sum(UA) / 8)):
            maxACC = WA + sum(UA) / 8
            torch.save(model.state_dict(), MODEL_PATH)
            # pd_matrix=pd.DataFrame(matrix)
            # writer=pd.ExcelWriter('test-result/seed-{}_epoch-{}.xlsx'.format(SEED,epoch))
            # pd_matrix.to_excel(writer,'page_1',float_format='%.5f')
            # writer.save()
            print('saving model,epoch:{},WA:{},UA:{}'.format(epoch, WA, sum(UA) / 8))
            logging.info('saving model,epoch:{},WA:{},UA:{}'.format(epoch, WA, sum(UA) / 8))
        print('Acc: {:.6f}\nUA:{},{}\nmaxWA:{},maxUA{}'.format(WA, UA, sum(UA) / 8, maxWA, maxUA))

        print(matrix)

