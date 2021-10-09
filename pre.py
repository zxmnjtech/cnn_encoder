import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, glob
import librosa
import librosa.display
import torch.optim as optim
import IPython
from IPython.display import Audio
from IPython.display import Image
import warnings; warnings.filterwarnings('ignore') #matplot lib complains about librosa

sample_rate = 48000

def feature_melspectrogram(
        waveform,
        sample_rate,
        fft=1024,
        winlen=512,
        window='hamming',
        hop=256,
        mels=128,
):
    # Produce the mel spectrogram for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
    # Using 8khz as upper frequency bound should be enough for most speech classification tasks
    melspectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=fft,
        win_length=winlen,
        window=window,
        hop_length=hop,
        n_mels=mels,
        fmax=sample_rate / 2)

    # convert from power (amplitude**2) to decibels
    # necessary for network to learn - doesn't converge with raw power spectrograms
    melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)

    return melspectrogram


def feature_mfcc(
        waveform,
        sample_rate,
        n_mfcc=40,
        fft=1024,
        winlen=512,
        window='hamming',
        # hop=256, # increases # of time steps; was not helpful
        mels=128
):
    # Compute the MFCCs for all STFT frames
    # 40 mel filterbanks (n_mfcc) = 40 coefficients
    mfc_coefficients = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=fft,
        win_length=winlen,
        window=window,
        # hop_length=hop,
        n_mels=mels,
        fmax=sample_rate / 2
    )

    return mfc_coefficients

def get_features(waveforms, features, samplerate):
    # initialize counter to track progress
    file_count = 0

    # process each waveform individually to get its MFCCs
    for waveform in waveforms:
        mfccs = feature_mfcc(waveform, sample_rate)
        features.append(mfccs)
        file_count += 1
        # print progress
        print('\r' + f' Processed {file_count}/{len(waveforms)} waveforms', end='')

    # return all features from list of waveforms
    return features
# Obtain audio data
def get_waveforms(file):
    waveform, _ = librosa.load(file, duration=2, sr=sample_rate)
    # , offset = 0.5
    # make sure waveform vectors are homogenous by defining explicitly
    waveform_homo = np.zeros((int(sample_rate * 2, )))
    waveform_homo[:len(waveform)] = waveform
    # return a single file's waveform
    return waveform_homo


# RAVDESS dataset emotions
# shift emotions left to be 0 indexed for PyTorch
# We only used audio data with 4 emotion tags
emotions_dictt = {
    # '0': 'surprised',
    '1': 'neutral',
    # '2': 'calm',
    '3': 'happy',
    '4': 'sad',
    '5': 'angry',
    # '6': 'fearful',
    # '7': 'disgust'
}

emotions_dict = {
    '01': "0",
    # '02': 'frustration',
    # '03': 'happy',
    '03': "1",
    '04': "2",
    # '06': 'fearful',
    '05': "3",  # excitement->happy
    # '08': 'surprised'
}

# Audio data storage location
data_path ="RAVDESS/Actor_*/*.wav"

# Load data
def load_data():
    emotions = []
    waveforms = []
    file_count = 0
    for file in glob.glob(data_path):
        file_name = os.path.basename(file)
        label = str(file_name.split("-")[2])
        if (label not in emotions_dict):
            continue
        emotion = int(emotions_dict[file_name.split("-")[2]])
        waveform = get_waveforms(file)
        waveforms.append(waveform)
        emotions.append(emotion)
        file_count += 1
        print('\r' + f' Processed {file_count}/{2659} audio samples', end='')
    return waveforms, emotions

waveforms, emotions = [],[]
waveforms, emotions = load_data()

# Create storage for training, validation, test sets and their indexes
train_set, valid_set, test_set = [], [], []
X_train, X_valid, X_test = [], [], []
y_train, y_valid, y_test = [], [], []

waveforms = np.array(waveforms)

# Process the sentiment of each label
for emotion_num in range(len(emotions_dictt)):
    # Save all the indexes corresponding to each emotion tag
    emotion_indices = [index for index, emotion in enumerate(emotions) if emotion == emotion_num]
    np.random.seed(69)
    emotion_indices = np.random.permutation(emotion_indices)
    dim = len(emotion_indices)
    train_indices = emotion_indices[:int(0.8 * dim)]
    valid_indices = emotion_indices[int(0.8 * dim):int(0.9 * dim)]
    test_indices = emotion_indices[int(0.9 * dim):]

    
    X_train.append(waveforms[train_indices, :])
    y_train.append(np.array([emotion_num] * len(train_indices), dtype=np.int32))
    X_valid.append(waveforms[valid_indices, :])
    y_valid.append(np.array([emotion_num] * len(valid_indices), dtype=np.int32))
    X_test.append(waveforms[test_indices, :])
    y_test.append(np.array([emotion_num] * len(test_indices), dtype=np.int32))

# Store the index of each sentiment set to verify the uniqueness between sets
    train_set.append(train_indices)
    valid_set.append(valid_indices)
    test_set.append(test_indices)

X_train = np.concatenate(X_train, axis=0)
X_valid = np.concatenate(X_valid, axis=0)
X_test = np.concatenate(X_test, axis=0)
y_train = np.concatenate(y_train, axis=0)
y_valid = np.concatenate(y_valid, axis=0)
y_test = np.concatenate(y_test, axis=0)

# combine and store indices for all emotions' train, validation, test sets to verify uniqueness of sets
train_set = np.concatenate(train_set, axis=0)
valid_set = np.concatenate(valid_set, axis=0)
test_set = np.concatenate(test_set, axis=0)

# check shape of each set
print(f'Training waveforms:{X_train.shape}, y_train:{y_train.shape}')
print(f'Validation waveforms:{X_valid.shape}, y_valid:{y_valid.shape}')
print(f'Test waveforms:{X_test.shape}, y_test:{y_test.shape}')

# Ensure that training, validation, and test sets are not overlapping/unique
# Get all unique indexes in all collections and the number of occurrences (count) of each index
uniques, count = np.unique(np.concatenate([train_set, test_set, valid_set], axis=0), return_counts=True)

# If each index appears only once, all collections are unique
if sum(count == 1) == len(emotions):
    print(f'\nSets are unique: {sum(count == 1)} samples out of {len(emotions)} are unique')
else:
    print(f'\nSets are NOT unique: {sum(count == 1)} samples out of {len(emotions)} are unique')

# Initialize the feature matrix
# We extract MFCC features from waveforms and store in respective 'features' array
features_train, features_valid, features_test = [],[],[]

print('Train waveforms:') # get training set features
features_train = get_features(X_train, features_train, sample_rate)

print('\n\nValidation waveforms:') # get validation set features
features_valid = get_features(X_valid, features_valid, sample_rate)

print('\n\nTest waveforms:') # get test set features
features_test = get_features(X_test, features_test, sample_rate)

print(f'\n\nFeatures set: {len(features_train)+len(features_test)+len(features_valid)} total, {len(features_train)} train, {len(features_valid)} validation, {len(features_test)} test samples')
print(f'Features (MFC coefficient matrix) shape: {len(features_train[0])} mel frequency coefficients x {len(features_train[0][1])} time steps')
# data augmentation
def awgn_augmentation(waveform, multiples=2, bits=16, snr_min=15, snr_max=30):
    # get length of waveform (should be 3*48k = 144k)
    wave_len = len(waveform)

    # Generate normally distributed (Gaussian) noises
    # one for each waveform and multiple (i.e. wave_len*multiples noises)
    noise = np.random.normal(size=(multiples, wave_len))

    # Normalize waveform and noise
    norm_constant = 2.0 ** (bits - 1)
    norm_wave = waveform / norm_constant
    norm_noise = noise / norm_constant

    # Compute power of waveform and power of noise
    signal_power = np.sum(norm_wave ** 2) / wave_len
    noise_power = np.sum(norm_noise ** 2, axis=1) / wave_len

    # Choose random SNR in decibels in range [15,30]
    snr = np.random.randint(snr_min, snr_max)

    # Apply whitening transformation: make the Gaussian noise into Gaussian white noise
    # Compute the covariance matrix used to whiten each noise
    # actual SNR = signal/noise (power)
    # actual noise power = 10**(-snr/10)
    covariance = np.sqrt((signal_power / noise_power) * 10 ** (- snr / 10))
    # Get covariance matrix with dim: (144000, 2) so we can transform 2 noises: dim (2, 144000)
    covariance = np.ones((wave_len, multiples)) * covariance

    # Since covariance and noise are arrays, * is the haddamard product
    # Take Haddamard product of covariance and noise to generate white noise
    multiple_augmented_waveforms = waveform + covariance.T * noise

    return multiple_augmented_waveforms

# data augmentation
def augment_waveforms(waveforms, features, emotions, multiples):
    # keep track of how many waveforms we've processed so we can add correct emotion label in the same order
    emotion_count = 0
    # keep track of how many augmented samples we've added
    added_count = 0
    # convert emotion array to list for more efficient appending
    emotions = emotions.tolist()

    for waveform in waveforms:

        augmented_waveforms = awgn_augmentation(waveform, multiples=multiples)

        # compute spectrogram for each of 2 augmented waveforms
        for augmented_waveform in augmented_waveforms:
            # Compute MFCCs over augmented waveforms
            augmented_mfcc = feature_mfcc(augmented_waveform, sample_rate=sample_rate)

            # append the augmented spectrogram to the rest of the native data
            features.append(augmented_mfcc)
            emotions.append(emotions[emotion_count])

            # keep track of new augmented samples
            added_count += 1

            # check progress
            print(
                '\r' + f'Processed {emotion_count + 1}/{len(waveforms)} waveforms for {added_count}/{len(waveforms) * multiples} new augmented samples',
                end='')

        # keep track of the emotion labels to append in order
        emotion_count += 1

        # store augmented waveforms to check their shape
        augmented_waveforms_temp.append(augmented_waveforms)

    return features, emotions

# store augmented waveforms to verify their shape and random-ness
augmented_waveforms_temp = []

# specify multiples of our dataset to add as augmented data
multiples = 2

print('Train waveforms:') # augment waveforms of training set
features_train , y_train = augment_waveforms(X_train, features_train, y_train, multiples)

print('\n\nValidation waveforms:') # augment waveforms of validation set
features_valid, y_valid = augment_waveforms(X_valid, features_valid, y_valid, multiples)

print('\n\nTest waveforms:') # augment waveforms of test set
features_test, y_test = augment_waveforms(X_test, features_test, y_test, multiples)

# The purpose of this is to add a dimension at the very beginning to make the input form conform to the CNN input format
X_train = np.expand_dims(features_train,1)
X_valid = np.expand_dims(features_valid, 1)
X_test = np.expand_dims(features_test,1)

# convert emotion labels from list back to numpy arrays for PyTorch to work with
y_train = np.array(y_train)
y_valid = np.array(y_valid)
y_test = np.array(y_test)

# confiorm that we have tensor-ready 4D data array
# should print (batch, channel, width, height) == (4320, 1, 128, 282) when multiples==2
print(f'Shape of 4D feature array for input tensor: {X_train.shape} train, {X_valid.shape} validation, {X_test.shape} test')
print(f'Shape of emotion labels: {y_train.shape} train, {y_valid.shape} validation, {y_test.shape} test')

# free up some RAM - no longer need full feature set or any waveforms
del features_train, features_valid, features_test, waveforms, augmented_waveforms_temp

###### SAVE #########
# choose save file name
filename = 'features+labels.npy'

# open file in write mode and write data
with open(filename, 'wb') as f:
    np.save(f, X_train)
    np.save(f, X_valid)
    np.save(f, X_test)
    np.save(f, y_train)
    np.save(f, y_valid)
    np.save(f, y_test)

print(f'Features and labels saved to {filename}')



##### LOAD #########
# choose load file name
filename = 'features+labels.npy'
MODEL_PATH = 'models/augment_6.pth'
# open file in read mode and read data
with open(filename, 'rb') as f:
    X_train = np.load(f)
    X_valid = np.load(f)
    X_test = np.load(f)
    y_train = np.load(f)
    y_valid = np.load(f)
    y_test = np.load(f)

# Check that we've recovered the right data
print(f'X_train:{X_train.shape}, y_train:{y_train.shape}')
print(f'X_valid:{X_valid.shape}, y_valid:{y_valid.shape}')
print(f'X_test:{X_test.shape}, y_test:{y_test.shape}')

class parallel_all_you_want(nn.Module):
    # Define all layers present in the network
    def __init__(self, num_emotions):
        super().__init__()

        ################ TRANSFORMER BLOCK #############################
        # maxpool the input feature map/tensor to the transformer
        # a rectangular kernel worked better here for the rectangular input spectrogram feature map/tensor
        self.transformer_maxpool = nn.MaxPool2d(kernel_size=[1, 4], stride=[1, 4])

        # define single transformer encoder layer
        # self-attention + feedforward network from "Attention is All You Need" paper
        # 4 multi-head self-attention layers each with 40-->512--->40 feedforward network
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=40,  # input feature (frequency) dim after maxpooling 40*282 -> 40*70 (MFC*time)
            nhead=4,  # 4 self-attention layers in each multi-head self-attention layer in each encoder block
            dim_feedforward=512,  # 2 linear layers in each encoder block's feedforward network: dim 40-->512--->40
            dropout=0.5,
            activation='relu'  # ReLU: avoid saturation/tame gradient/reduce compute time
        )
=
        # I'm using 4 instead of the 6 identical stacked encoder layrs used in Attention is All You Need paper
        # Complete transformer block contains 4 full transformer encoder layers (each w/ multihead self-attention+feedforward)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=6)

        self.fc1_linear = nn.Linear(40, num_emotions)

        ### Softmax layer for the 8 output logits from final FC linear layer
        # When a sample passes through the softmax layer and outputs a vector, the index of the number with the largest value in the vector will be used as the predicted label of the sample
        self.softmax_out = nn.Softmax(dim=1)  # dim==1 is the freq embedding

    # define one complete parallel fwd pass of input feature tensor thru 2*conv+1*transformer blocks
    def forward(self, x):
        x_maxpool = self.transformer_maxpool(x)

        # remove channel dim: 1*40*70 --> 40*70
        x_maxpool_reduced = torch.squeeze(x_maxpool, 1)

        # convert maxpooled feature map format: batch * freq * time ---> time * batch * freq format
        # because transformer encoder layer requires tensor in format: time * batch * embedding (freq)
        x = x_maxpool_reduced.permute(2, 0, 1)

        # finally, pass reduced input feature map x into transformer encoder layers
        transformer_output = self.transformer_encoder(x)

        # create final feature emedding from transformer layer by taking mean in the time dimension (now the 0th dim)
        # transformer outputs 2x40 (MFCC embedding*time) feature map, take mean of columns i.e. take time average
        transformer_embedding = torch.mean(transformer_output, dim=0)  # dim 40x70 --> 40

        ############# concatenate freq embeddings from convolutional and transformer blocks ######
        # concatenate embedding tensors output by parallel 2*conv and 1*transformer blocks
        complete_embedding = torch.cat([transformer_embedding], dim=1)
        # , conv2d_embedding2
        # conv2d_embedding1,
        ######### final FC linear layer, need logits for loss #########################
        output_logits = self.fc1_linear(complete_embedding)

        ######### Final Softmax layer: use logits from FC linear, get softmax for prediction ######
        output_softmax = self.softmax_out(output_logits)

        # need output logits to compute cross entropy loss, need softmax probabilities to predict class
        return output_logits, output_softmax, complete_embedding

from torchsummary import summary

# need device to instantiate model
device = 'cuda'

# 4 instantiated emotion models and transferred to GPU
model = parallel_all_you_want(len(emotions_dict)).to(device)

# summary can print the network structure and parameters
# include input feature map dims in call to summary()
summary(model, input_size=(1,40,188))

# Define the loss function; CrossEntropyLoss() is quite standard for many types of problems
def criterion(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions, target=targets)

# Neural network optimizer is mainly to optimize our neural network, make it faster in our training process, and save the time of social network training.
optimizer = torch.optim.SGD(model.parameters(),lr=0.01, weight_decay=1e-3, momentum=0.8)

# Define the function to create a single step of the training phase
def make_train_step(model, criterion, optimizer):
    # Define the training steps of the training phase
    def train_step(X, Y):
        output_logits, output_softmax,temp = model(X)
        # Different values of dim indicate different dimensions. In particular, dim=0 indicates a two-dimensional column, and dim=1 indicates a row in a two-dimensional matrix.
        predictions = torch.argmax(output_softmax, dim=1)
        accuracy = torch.sum(Y == predictions) / float(len(Y))

        # compute loss on logits because nn.CrossEntropyLoss implements log softmax
        loss = criterion(output_logits, Y)

        # compute gradients for the optimizer to use
        loss.backward()

        # update network parameters based on gradient stored (by calling loss.backward())
        optimizer.step()

        # zero out gradients for next pass
        # pytorch accumulates gradients from backwards passes (convenient for RNNs)
        optimizer.zero_grad()

        return loss.item(), accuracy * 100

    return train_step


def make_validate_fnc(model, criterion):
    def validate(X, Y):
        with torch.no_grad():
            model.eval()

            # Obtain model predictions on the validation set
            output_logits, output_softmax,temp = model(X)
            predictions = torch.argmax(output_softmax, dim=1)

            # calculate the mean accuracy over the entire validation set
            accuracy = torch.sum(Y == predictions) / float(len(Y))

            # compute error from logits (nn.crossentropy implements softmax)
            loss = criterion(output_logits, Y)

        return loss.item(), accuracy * 100, predictions

    return validate

def make_save_checkpoint():
    def save_checkpoint(optimizer, model, epoch, filename):
        checkpoint_dict = {
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint_dict, filename)
    return save_checkpoint

def load_checkpoint(optimizer, model, filename):
    checkpoint_dict = torch.load(filename)
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return epoch


# Get the size of the training set to calculate # iterations and minibatch index The size of the first dimension is
train_size = X_train.shape[0]

# Select small batches (always 32
minibatch = 32

# set device to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'{device} selected')

# Instantiate the model and move it to the GPU for training
model = parallel_all_you_want(num_emotions=len(emotions_dict)).to(device)
print('Number of trainable params: ', sum(p.numel() for p in model.parameters()))

# encountered bugs in google colab only, unless I explicitly defined optimizer in this cell...
optimizer = torch.optim.SGD(model.parameters(),lr=0.01, weight_decay=1e-3, momentum=0.8)

# instantiate the checkpoint save function
save_checkpoint = make_save_checkpoint()

# instantiate the training step function
train_step = make_train_step(model, criterion, optimizer=optimizer)

# instantiate the validation loop function
validate = make_validate_fnc(model, criterion)

train_losses = []
valid_losses = []
acc_sum= []

# create training loop for one complete epoch (entire training set)
def train(optimizer, model, num_epochs, X_train, Y_train, X_valid, Y_valid):
    for epoch in range(num_epochs):

        # set model to train phase
        model.train()

        train_indices = np.random.permutation(train_size)
        X_train = X_train[train_indices, :, :, :]
        Y_train = Y_train[train_indices]
        epoch_acc = 0
        epoch_loss = 0
        num_iterations = int(train_size / minibatch)

        # create a loop for each minibatch of 32 samples:
        for i in range(num_iterations):
            # we have to track and update minibatch position for the current minibatch
            # if we take a random batch position from a set, we almost certainly will skip some of the data in that set
            # track minibatch position based on iteration number:
            batch_start = i * minibatch
            # ensure we don't go out of the bounds of our training set:
            batch_end = min(batch_start + minibatch, train_size)
            # ensure we don't have an index error
            actual_batch_size = batch_end - batch_start

            # get training minibatch with all channnels and 2D feature dims
            X = X_train[batch_start:batch_end, :, :, :]
            # get training minibatch labels
            Y = Y_train[batch_start:batch_end]

            # instantiate training tensors
            X_tensor = torch.tensor(X, device=device).float()
            Y_tensor = torch.tensor(Y, dtype=torch.long, device=device)

            # Pass input tensors thru 1 training step (fwd+backwards pass)
            loss, acc = train_step(X_tensor, Y_tensor)

            # aggregate batch accuracy to measure progress of entire epoch
            epoch_acc += acc * actual_batch_size / train_size
            epoch_loss += loss * actual_batch_size / train_size

            # keep track of the iteration to see if the model's too slow
            print('\r' + f'Epoch {epoch}: iteration {i}/{num_iterations}', end='')

        # create tensors from validation set
        X_valid_tensor = torch.tensor(X_valid, device=device).float()
        Y_valid_tensor = torch.tensor(Y_valid, dtype=torch.long, device=device)

        # calculate validation metrics to keep track of progress; don't need predictions now
        valid_loss, valid_acc, _ = validate(X_valid_tensor, Y_valid_tensor)

        # accumulate scalar performance metrics at each epoch to track and plot later
        train_losses.append(epoch_loss)
        valid_losses.append(valid_loss)
        acc_sum.append(valid_acc)
        # Save checkpoint of the model
        # checkpoint_filename = './models/checkpoints/parallel_all_you_wantFINAL-{:03d}.pkl'.format(
        #     epoch)
        # save_checkpoint(optimizer, model, epoch, checkpoint_filename)
        torch.save(model.state_dict(), MODEL_PATH)
        # keep track of each epoch's progress
        print(
            f'\nEpoch {epoch} --- loss:{epoch_loss:.3f}, Epoch accuracy:{epoch_acc:.2f}%, Validation loss:{valid_loss:.3f}, Validation accuracy:{valid_acc:.2f}%')

# choose number of epochs higher than reasonable so we can manually stop training
num_epochs = 200

# train it!
train(optimizer, model, num_epochs, X_train, y_train, X_valid, y_valid)
total = 0
for ele in range(0, len(acc_sum)):
    total = total + acc_sum[ele]
print("Average Accuracy: ", total/len(acc_sum))

plt.title('Loss Curve for Parallel is All You Want Model')
plt.ylabel('Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=16)
plt.plot(train_losses[:],'b')
plt.plot(valid_losses[:],'r')
plt.legend(['Training loss','Validation loss'])
plt.show()
