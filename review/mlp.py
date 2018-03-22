import gc

import keras
import keras.preprocessing.sequence
import keras.preprocessing.text
from keras import backend as K
from keras.layers import Dense, Input
from sklearn.pipeline import make_union

from util import create_basic_feature_extractor, create_BoW_feature_extractor, run_model


def build_mlp_model(input_dim):
    model_input = Input(shape=(input_dim,), sparse=True)
    x = Dense(192, activation='relu')(model_input)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1)(x)

    model = keras.Model(model_input, output)
    model.compile(
        loss='mean_squared_error',
        optimizer=keras.optimizers.Adam(lr=3e-3)
    )
    return model


def mlp_predict(X_train, y_train, X_test):
    model = build_mlp_model(input_dim=X_train.shape[1])
    model.fit(
        X_train, y_train,
        batch_size=2048,
        epochs=3,
        verbose=True
    )
    pred = model.predict(X_test, batch_size=2048).ravel()

    del model
    gc.collect()
    K.clear_session()
    return pred


def run_mlp(train, test, y_train, num_models):
    feature_extractor = make_union(create_basic_feature_extractor(), create_BoW_feature_extractor())
    X_train = feature_extractor.fit_transform(train)
    X_test = feature_extractor.transform(test)
    return [mlp_predict(X_train, y_train, X_test) for _ in range(num_models)]


if __name__ == '__main__':
    run_model(model_runner=run_mlp, model_name='MLP')

