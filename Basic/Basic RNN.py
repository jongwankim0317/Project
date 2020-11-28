import tensorflow as tf
import os

EPOCHS = 100
NUM_WORDS = 10000
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.emb = tf.keras.layers.Embedding(NUM_WORDS, 16)
        self.rnn = tf.keras.layers.LSTM(32)
        self.dense = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, x, training=None, mask=None):
        x = self.emb(x)
        x = self.rnn(x)
        return self.dense(x)

@tf.function
def train_step(model, inputs, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        prediction = model(inputs, training=True)
        loss = loss_object(labels, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, prediction)

@tf.function
def test_step(model, inputs, labels, loss_object, test_loss, test_accuracy):
    prediction = model(inputs, training = False)

    t_loss = loss_object(labels, prediction)
    test_loss(t_loss)
    test_accuracy(labels, prediction)


#Preparing dataset
imdb = tf.keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = NUM_WORDS)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train,
                                                        value=0,
                                                        padding='pre',
                                                        maxlen=32)

x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test,
                                                        value=0,
                                                        padding='pre',
                                                        maxlen=32)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(64)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)

# Create Model
model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# Learning Loop

for epoch in range(EPOCHS):
    for train_seqs, train_labels in train_ds:
        train_step(model, train_seqs, train_labels, loss_object, optimizer, train_loss, train_accuracy)

    for test_seqs, test_labels in test_ds:
        test_step(model, test_seqs, test_labels, loss_object, test_loss, test_accuracy)

    template = 'Epoch: {}, Loss: {}, Accuracy: {}, Test loss: {}, Test accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))

    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
