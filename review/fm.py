import random

from fastFM.als import FMRegression
from joblib import Parallel, delayed
from sklearn.pipeline import make_union

from util import create_basic_feature_extractor, create_BoW_feature_extractor, run_model


def fm_predict(X_train, y_train, X_test):
    fm = FMRegression(init_stdev=0.0001, rank=128, l2_reg_w=20, l2_reg_V=400,
                      n_iter=7, random_state=random.randint(0, 1000))
    fm.fit(X_train, y_train)
    return fm.predict(X_test)


def run_fm(train, test, y_train, num_models):
    feature_extractor = make_union(create_basic_feature_extractor(), create_BoW_feature_extractor())
    X_train = feature_extractor.fit_transform(train)
    X_test = feature_extractor.transform(test)
    return Parallel(n_jobs=-1, max_nbytes=None)(
        delayed(fm_predict)(X_train, y_train, X_test) for _ in range(num_models)
    )


if __name__ == '__main__':
    run_model(model_runner=run_fm, model_name='FM')

