from sklearn.linear_model import Ridge
from sklearn.pipeline import make_union

from util import create_basic_feature_extractor, create_BoW_feature_extractor, run_model


def lr_predict(X_train, y_train, X_test):
    ridge = Ridge(alpha=15.0, solver='sag', max_iter=100)
    ridge.fit(X_train, y_train)
    return ridge.predict(X_test)


def run_lr(train, test, y_train, num_models):
    feature_extractor = make_union(create_basic_feature_extractor(), create_BoW_feature_extractor())
    X_train = feature_extractor.fit_transform(train)
    X_test = feature_extractor.transform(test)
    return [lr_predict(X_train, y_train, X_test) for _ in range(num_models)]


if __name__ == '__main__':
    run_model(model_runner=run_lr, model_name='LR')

