import pandas as pd
import h5py as h5
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout, Input, Lambda
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from matplotlib import pyplot as plt
from keras.regularizers import l2, l1_l2

# Parameters and options
DROP_OUT = True
DROP_OUT_1 = 0.5
DROP_OUT_2 = 0.5
DROP_OUT_3 = 0.5

PERCENTAGE_TRAIN_DATASET = 0.85
PERCENTAGE_PREDICTION = 0.01
MULTIPLY_WEIGHTS = False

FILTERS = 16  # Reduced number of filters to simplify the model
KERNEL_SIZE = 3
INPUT_SHAPE = (18970, 1)
STRIDES = 1

POOL_SIZE = 2

NR_NEURONS_1 = 128  # Reduced number of neurons
NR_NEURONS_2 = 64
NR_NEURONS_OUT = 1

NR_EPOCHS = 50
BATCH_SIZE = 32  # Adjusted batch size

MODEL_FILE_OUT_NAME = r'C:\Users\guimo\OneDrive\Desktop\Informática\1CNN'

# Load data
try:
    df3 = pd.read_csv(r"C:\Users\guimo\OneDrive\Desktop\Informática\1CNN\subjects.csv", index_col=0)
    df4 = pd.read_csv(r"C:\Users\guimo\OneDrive\Desktop\Informática\1CNN\connection_weights_example.csv", index_col=0)
    ds_weights = df4['weights_1']

    data5 = h5.File(r"C:\Users\guimo\OneDrive\Desktop\Informática\1CNN\genotype.h5", "r")
    dset1 = data5['data']
    dfa = pd.DataFrame(np.array(dset1))
except Exception as e:
    print(f"Error loading data: {e}")

# Apply weights if needed
if MULTIPLY_WEIGHTS:
    dfa = dfa.astype(float).mul(ds_weights, axis=1)
else:
    dfa = dfa / 2.0

# Normalize data
dfa = (dfa - dfa.mean()) / dfa.std()

# Combine data and labels
dfb = pd.concat([dfa, df3['labels']], axis=1)

df_train = dfb.sample(frac=PERCENTAGE_TRAIN_DATASET)
df_train_labels = df_train['labels']
df_train = df_train.drop(['labels'], axis=1)

df_test = dfb.loc[~dfb.index.isin(df_train.index)]
df_test_labels = df_test['labels']
df_test = df_test.drop(['labels'], axis=1)

df_predict = dfb.sample(frac=PERCENTAGE_PREDICTION)
df_predict_labels = df_predict['labels']
df_predict = df_predict.drop(['labels'], axis=1)

print(f'df_train.shape = {df_train.shape}; df_test.shape = {df_test.shape}')

# Reshape data for Conv1D
df_train = df_train.values.reshape(-1, INPUT_SHAPE[0], 1)
df_test = df_test.values.reshape(-1, INPUT_SHAPE[0], 1)
df_predict = df_predict.values.reshape(-1, INPUT_SHAPE[0], 1)

# Define model input
input_layer = Input(shape=INPUT_SHAPE)

# First Conv1D Layer
x = Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, strides=STRIDES, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(input_layer)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=POOL_SIZE)(x)
if DROP_OUT:
    x = Dropout(DROP_OUT_1)(x)

# Second Conv1D Layer
x = Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, strides=STRIDES, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=POOL_SIZE)(x)
if DROP_OUT:
    x = Dropout(DROP_OUT_2)(x)

# Flatten and Dense Layers
x = Flatten()(x)
x = Dense(NR_NEURONS_1, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
x = BatchNormalization()(x)
if DROP_OUT:
    x = Dropout(DROP_OUT_3)(x)

# Output Layer
output_layer = Dense(NR_NEURONS_OUT, activation='sigmoid')(x)

# Create model
model = Model(inputs=input_layer, outputs=output_layer)

# Optimizer with learning rate scheduling
initial_lr = 1e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_lr, decay_steps=10000, decay_rate=0.9, staircase=True)

optimizer = Adam(learning_rate=lr_schedule)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(model.summary())

# Callbacks for early stopping and learning rate reduction
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('best_model.keras', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)

# Train model
history = model.fit(df_train, 
                    df_train_labels, 
                    validation_data=(df_test, df_test_labels), 
                    epochs=NR_EPOCHS, batch_size=BATCH_SIZE, verbose=1, 
                    callbacks=[es, mc, rlrp])

# Load the best model and evaluate
saved_model = load_model('best_model.keras')

_, train_accuracy = saved_model.evaluate(df_train, df_train_labels)
_, test_accuracy = saved_model.evaluate(df_test, df_test_labels)
print('Train Accuracy: %.2f' % (train_accuracy * 100))
print('Test Accuracy: %.2f' % (test_accuracy * 100))

# Predictions
predictions = (saved_model.predict(df_predict) > 0.5).astype(int)

labels_values = df_predict_labels.values
nr_correct_estimations = 0
nr_wrong_estimations = 0

for i in range(df_predict.shape[0]):
    if predictions[i][0] != labels_values[i]:
        nr_wrong_estimations += 1
    else:
        nr_correct_estimations += 1

print(f'Number of correct estimations = {nr_correct_estimations} | Number of incorrect estimations = {nr_wrong_estimations} | Percentage of incorrect estimations {nr_wrong_estimations / (nr_correct_estimations + nr_wrong_estimations)}')

# Plot training history
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()

plt.subplot(122)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()