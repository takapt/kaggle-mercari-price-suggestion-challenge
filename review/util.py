from operator import itemgetter

import keras
import keras.preprocessing.sequence
import keras.preprocessing.text
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing import LabelBinarizer


def rmse(y, correct_y):
    return np.sqrt(mean_squared_error(y, correct_y))


def split_category_name(category_name: str):
    spl = category_name.split('/', 2)
    if len(spl) == 3:
        return spl
    else:
        return ['', '', '']


def add_sub_cagegories(dataset):
    spl = dataset['category_name'].apply(split_category_name)
    dataset['category1'], dataset['category2'], dataset['category3'] = zip(*spl)


def preprocess(df):
    df['log1p_price'] = np.log1p(df['price'])

    df['name'].fillna('', inplace=True)
    df['category_name'].fillna('', inplace=True)
    df['brand_name'].fillna('', inplace=True)
    df['item_description'].fillna('', inplace=True)

    add_sub_cagegories(df)

    # This preprocess improves score significantly.
    #
    # For example, when selling "iPhone",
    # "iPhone" must be in 'name' and "iPhone" should be in 'item_description'.
    # However, some 'item_description' could not have "iPhone".
    # In such cases, models cannot capture the feature "iPhone".
    # This preprocess is necessary for avoiding such cases.
    df['name'] = df['name'] + ' ' + df['brand_name']
    df['item_description'] = df['item_description'] + ' ' + df['name']


class LabelBinarizerForPipeline(LabelBinarizer):
    """
    This class is a wrapper for pipeline.
    LabelBinarizer cannot be used for pipeline due to its fit/transform/fit_transform APIs.
    """

    def fit(self, X, y=None):
        return super(LabelBinarizerForPipeline, self).fit(X)

    def transform(self, X, y=None):
        return super(LabelBinarizerForPipeline, self).transform(X)

    def fit_transform(self, X, y=None):
        return super(LabelBinarizerForPipeline, self).fit(X).transform(X)


def create_column_extractor(column, extractor):
    """
    This method can be used to pass a DataFrame to a feature extractor.

    Example

    column_extractor = create_column_extractor('brand_name', LabelBinarizerForPipeline(sparse_output=True))
    column_extractor.fit(train)

    is the same process as

    X = train['brand_name']
    lb = LabelBinarizerForPipeline(sparse_output=True)
    lb.fit(X)

    :param column: a column name of DataFrame for feature extraction
    :param extractor: a feature extractor
    :return:
    """
    return make_pipeline(
        FunctionTransformer(itemgetter(column), validate=False),
        extractor
    )


def create_basic_feature_extractor():
    return make_union(
        create_column_extractor('item_condition_id', LabelBinarizerForPipeline(sparse_output=True)),
        create_column_extractor('shipping', LabelBinarizerForPipeline(sparse_output=True)),
        create_column_extractor('brand_name', LabelBinarizerForPipeline(sparse_output=True)),
        create_column_extractor('category1', LabelBinarizerForPipeline(sparse_output=True)),
        create_column_extractor('category2', LabelBinarizerForPipeline(sparse_output=True)),
        create_column_extractor('category3', LabelBinarizerForPipeline(sparse_output=True))
    )


def create_BoW_feature_extractor():
    return make_union(
        create_column_extractor(
            'name',
            CountVectorizer(token_pattern='\w+', ngram_range=(1, 2), max_features=100000, binary=True)
        ),
        create_column_extractor(
            'item_description',
            CountVectorizer(token_pattern='\w+', ngram_range=(1, 2), max_features=100000, binary=True)
        )
    )


MAX_LEN_NAME = 20
MAX_LEN_DESC = 160
EMBEDDING_SIZE = 128


def create_keras_seqs(texts_train, texts_test, maxlen):
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(texts_train)

    seqs_train = keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(texts_train), maxlen=maxlen,
                                                            truncating='post')
    seqs_test = keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(texts_test), maxlen=maxlen,
                                                           truncating='post')
    return seqs_train, seqs_test


def create_keras_input(train, test):
    basic_feature_extractor = create_basic_feature_extractor()
    basic_input_train = basic_feature_extractor.fit_transform(train)
    basic_input_test = basic_feature_extractor.transform(test)

    name_seqs_train, name_seqs_test = create_keras_seqs(train['name'], test['name'], maxlen=MAX_LEN_NAME)
    desc_seqs_train, desc_seqs_test = create_keras_seqs(train['item_description'], test['item_description'], maxlen=MAX_LEN_DESC)

    input_train = {
        'basic': basic_input_train,
        'name': name_seqs_train,
        'item_description': desc_seqs_train
    }
    input_test = {
        'basic': basic_input_test,
        'name': name_seqs_test,
        'item_description': desc_seqs_test
    }
    return input_train, input_test


def load_train_test():
    dataset = pd.read_table('../input/train.tsv')
    dataset = dataset[dataset.price > 0].reset_index(drop=True)  # Drop dirty data
    train, test = train_test_split(dataset, test_size=0.05, random_state=114514)
    train = train.reset_index(drop=True).copy()
    test = test.reset_index(drop=True).copy()

    preprocess(train)
    preprocess(test)

    return train, test


def run_model(model_runner, model_name):
    num_models = 8

    print('Model:', model_name)

    train, test = load_train_test()

    y_train = train['log1p_price'].values
    y_test = test['log1p_price'].values
    y_scaler = StandardScaler()
    scaled_y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    preds = model_runner(train, test, scaled_y_train, num_models)
    preds = [y_scaler.inverse_transform(pred) for pred in preds]

    print('losses:', [rmse(pred, y_test) for pred in preds])
    print('averaging loss:', rmse(np.mean(preds, axis=0), y_test))

    pred_df = pd.DataFrame()
    pred_df['train_id'] = test['train_id']
    for i, pred in enumerate(preds):
        pred_df[model_name + '{}'.format(i)] = pred
    filename_to_save = '../output/' + model_name + '_pred.csv'
    pred_df.to_csv(filename_to_save, index=False)
    print('Saved preds into', filename_to_save)
