import gc

import keras
import keras.preprocessing.sequence
import keras.preprocessing.text
from keras import backend as K
from keras.layers import Dense, Input, Embedding, CuDNNGRU

from util import MAX_LEN_NAME, MAX_LEN_DESC, EMBEDDING_SIZE, create_keras_input, run_model


def build_rnn_model(basic_input_dim, vocabulary_size_name, vocabulary_size_desc):
    basic_input = Input(shape=(basic_input_dim,), sparse=True, name='basic')

    input_name = Input(shape=(MAX_LEN_NAME,), name='name')
    embedding_name = Embedding(input_dim=vocabulary_size_name, output_dim=EMBEDDING_SIZE)(input_name)
    rnn_name = CuDNNGRU(256)(embedding_name)

    input_desc = Input(shape=(MAX_LEN_DESC,), name='item_description')
    embedding_desc = Embedding(input_dim=vocabulary_size_desc, output_dim=EMBEDDING_SIZE)(input_desc)
    rnn_desc = CuDNNGRU(512)(embedding_desc)

    x = Dense(32, activation='relu')(basic_input)
    x = keras.layers.concatenate([
        x,
        rnn_name,
        rnn_desc
    ])
    x = Dense(192, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1)(x)

    model = keras.Model(
        inputs=[
            basic_input,
            input_name,
            input_desc
        ],
        outputs=output
    )
    model.compile(
        loss='mean_squared_error',
        optimizer=keras.optimizers.Adam(lr=2e-3)
    )
    return model


def rnn_predict(keras_input_train, y_train, keras_input_test):
    model = build_rnn_model(
        basic_input_dim=keras_input_train['basic'].shape[1],
        vocabulary_size_name=keras_input_train['name'].max() + 1,
        vocabulary_size_desc=keras_input_train['item_description'].max() + 1
    )
    model.fit(
        keras_input_train, y_train,
        batch_size=1024,
        epochs=3,
        verbose=True
    )
    pred = model.predict(keras_input_test, batch_size=1024).ravel()

    del model
    gc.collect()
    K.clear_session()
    return pred


def run_rnn(train, test, y_train, num_models):
    keras_input_train, keras_input_test = create_keras_input(train, test)
    return [rnn_predict(keras_input_train, y_train, keras_input_test) for _ in range(num_models)]


if __name__ == '__main__':
    run_model(model_runner=run_rnn, model_name='RNN')
