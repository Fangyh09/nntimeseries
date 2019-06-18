# nntimeseries

The repository provides the code for the paper  [*Autoregressive Convolutional 
Neural Networks for Asynchronous Time Series*](https://arxiv.org/abs/1703.04122), as well as general code for running grid serach on keras models. 

Files 'nnts/models/{CNN, LSTM, LSTM2, LR, SOCNN}.py' provide code for testing 
respective models, with the last one implementing the proposed 
Significance-Offset CNN and LSTM2 implementing multi-layer LSTM.

**Basic Usage**

Each of the model files can be run as a script, e.g.
- `python ./CNN.py --dataset=artificial`   # default save file 
-	`python ./SOCNN.py --dataset=household --save_file=results/household_0.pkl`

Parameters for grid search can be specified in each of the above 
files. 

Each of these files defines a model class that can be imported and used on external dataset, as shown in example.ipynb file.

The repository supports optimization of the above models on artifical 
multivariate noisy AR time series and household electricity conspumption 
dataset
[https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption)
The dataset has to be specified alongside the paremeters in each of 
the files listed above. 

To generate aritficial datasets used in model evaluation in the paper, run 'python generate_artifical.py'.

**Requirements**
- python   >= 3.5.3
- Keras    >= 2.0.2
- numpy    >= 1.12.12
- pandas   >= 0.19.2
- h5py     >= 2.6.0

Feel free to contact Mikolaj Binkowski ('mikbinkowski at gmail.com') with any 
questions and issues.


##
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
inp (InputLayer)                 (None, 60, 65)        0
____________________________________________________________________________________________________
significance1 (Conv1D)           (None, 60, 8)         528
____________________________________________________________________________________________________
significance1BN (BatchNormalizat (None, 60, 8)         32
____________________________________________________________________________________________________
significance1act (LeakyReLU)     (None, 60, 8)         0
____________________________________________________________________________________________________
significance2 (Conv1D)           (None, 60, 8)         200
____________________________________________________________________________________________________
significance2BN (BatchNormalizat (None, 60, 8)         32
____________________________________________________________________________________________________
significance2act (LeakyReLU)     (None, 60, 8)         0
____________________________________________________________________________________________________
significance3 (Conv1D)           (None, 60, 8)         72
____________________________________________________________________________________________________
significance3BN (BatchNormalizat (None, 60, 8)         32
____________________________________________________________________________________________________
significance3act (LeakyReLU)     (None, 60, 8)         0
____________________________________________________________________________________________________
significance4 (Conv1D)           (None, 60, 8)         200
____________________________________________________________________________________________________
significance4BN (BatchNormalizat (None, 60, 8)         32
____________________________________________________________________________________________________
significance4act (LeakyReLU)     (None, 60, 8)         0
____________________________________________________________________________________________________
significance5 (Conv1D)           (None, 60, 8)         72
____________________________________________________________________________________________________
significance5BN (BatchNormalizat (None, 60, 8)         32
____________________________________________________________________________________________________
significance5act (LeakyReLU)     (None, 60, 8)         0
____________________________________________________________________________________________________
significance6 (Conv1D)           (None, 60, 8)         200
____________________________________________________________________________________________________
significance6BN (BatchNormalizat (None, 60, 8)         32
____________________________________________________________________________________________________
significance6act (LeakyReLU)     (None, 60, 8)         0
____________________________________________________________________________________________________
significance7 (Conv1D)           (None, 60, 8)         72
____________________________________________________________________________________________________
significance7BN (BatchNormalizat (None, 60, 8)         32
____________________________________________________________________________________________________
significance7act (LeakyReLU)     (None, 60, 8)         0
____________________________________________________________________________________________________
significance8 (Conv1D)           (None, 60, 8)         200
____________________________________________________________________________________________________
significance8BN (BatchNormalizat (None, 60, 8)         32
____________________________________________________________________________________________________
significance8act (LeakyReLU)     (None, 60, 8)         0
____________________________________________________________________________________________________
significance9 (Conv1D)           (None, 60, 8)         72
____________________________________________________________________________________________________
significance9BN (BatchNormalizat (None, 60, 8)         32
____________________________________________________________________________________________________
significance9act (LeakyReLU)     (None, 60, 8)         0
____________________________________________________________________________________________________
significance10 (Conv1D)          (None, 60, 65)        1625
____________________________________________________________________________________________________
offset1 (Conv1D)                 (None, 60, 65)        4290
____________________________________________________________________________________________________
significance10BN (BatchNormaliza (None, 60, 65)        260
____________________________________________________________________________________________________
offset1BN (BatchNormalization)   (None, 60, 65)        260
____________________________________________________________________________________________________
significance10act (LeakyReLU)    (None, 60, 65)        0
____________________________________________________________________________________________________
offset1act (LeakyReLU)           (None, 60, 65)        0
____________________________________________________________________________________________________
value_input (InputLayer)         (None, 60, 65)        0
____________________________________________________________________________________________________
permute_2 (Permute)              (None, 65, 60)        0
____________________________________________________________________________________________________
value_output (Add)               (None, 60, 65)        0
____________________________________________________________________________________________________
softmax (TimeDistributed)        (None, 65, 60)        0
____________________________________________________________________________________________________
permute_1 (Permute)              (None, 65, 60)        0
____________________________________________________________________________________________________
significancemerge (Multiply)     (None, 65, 60)        0
____________________________________________________________________________________________________
locally_connected1d_1 (LocallyCo (None, 65, 1)         3965
____________________________________________________________________________________________________
main_output (Permute)            (None, 1, 65)         0
====================================================================================================
Total params: 12,304.0
Trainable params: 11,900.0
Non-trainable params: 404.0
____________________________________________________________________________________________________
```
