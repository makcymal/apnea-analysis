import numpy as np
import pandas as pd
from preproc import fir_filter_coef
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from catboost import CatBoostClassifier


# all EEG records has frequency of 200 Hz
FREQ = 200
# FIR filter on 10 data taps, passing [5Hz, 40Hz] signal
FILT_COEF = fir_filter_coef(10, 5, 40, FREQ)

# patients without apnoe
HEALTHY_INDICES = set(list(range(1, 16)) + list(range(26, 31)))
# patients with apnoe
ILL_INDICES = set(range(1, 41)) - HEALTHY_INDICES


# crossvalidate back propagation perceptron with given params
def crossval_neunet(
    X, y, n_folds, arch, activ, out_activ, loss, optim, epochs, batch_size
):
    arch = [X.shape[1]] + arch + [1]
    kfold = StratifiedKFold(n_folds, shuffle=True)
    cvscores = []

    model_no = 0
    for train, test in kfold.split(X, y):
        model = keras.Sequential(
            [layers.Input([arch[0]])]
            + [
                layers.Dense(units=arch[i], activation=activ)
                for i in range(1, len(arch) - 1)
            ]
            + [layers.Dense(units=arch[-1], activation=out_activ)]
        )
        model.compile(
            loss=loss, optimizer=optim, metrics=["accuracy", "precision", "recall"]
        )
        model.fit(
            X.iloc[train],
            y.iloc[train],
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )
        model.save(f"model_{model_no}.keras")
        model_no += 1
        score = model.evaluate(X.iloc[test], y.iloc[test], verbose=0)
        cvscores.append(score[1:])

    return np.array(cvscores)


# crossvalide random forest
def crossval_forest(X, y, n_folds):
    kfold = StratifiedKFold(n_folds, shuffle=True)
    cvscores = []

    for train, test in kfold.split(X, y):
        model = RandomForestClassifier(
            verbose=False, max_features="log2", bootstrap=False, random_state=424
        )
        model.fit(X.iloc[train], y.iloc[train])
        y_pred = model.predict(X.iloc[test])
        cvscores.append(
            [
                accuracy_score(y.iloc[test], y_pred),
                precision_score(y.iloc[test], y_pred),
                recall_score(y.iloc[test], y_pred),
            ]
        )

    return np.array(cvscores)


# crossvalidate catboost forest
def crossval_catboost(X, y, n_folds):
    kfold = StratifiedKFold(n_folds, shuffle=True)
    cvscores = []

    for train, test in kfold.split(X, y):
        model = CatBoostClassifier(verbose=False, random_state=424)
        _ = model.fit(X.iloc[train], y.iloc[train])
        y_pred = model.predict(X.iloc[test])
        cvscores.append(
            [
                accuracy_score(y.iloc[test], y_pred),
                precision_score(y.iloc[test], y_pred),
                recall_score(y.iloc[test], y_pred),
            ]
        )

    return np.array(cvscores)


# model based on perceptron net
class ApnoeModel:
    def __init__(self):
        self.get_X_y()
        self.fit()

    # builds dataframe with:
    # - EEG O1, O2 rhythms percentage
    # - 3 principal components of common health indicators
    def get_X_y(self):
        df = pd.read_csv("rhythms.csv")
        eeg_features_O1 = ["delta_O1", "theta_O1", "alpha_O1", "beta_O1", "gamma_O1"]
        eeg_features_O2 = ["delta_O2", "theta_O2", "alpha_O2", "beta_O2", "gamma_O2"]
        eeg_features = eeg_features_O1 + eeg_features_O2

        dfull = pd.read_csv("full_data.csv")
        features = ["age", "sex", "height", "weight", "pulse", "BPsys", "BPdia", "ODI"]
        self.pca = PCA(n_components=3)
        self.pca.fit(dfull[features])
        X_pc = self.pca.transform(dfull[features])

        X = pd.concat(
            (df[eeg_features], pd.DataFrame(X_pc, columns=["pc1", "pc2", "pc3"])),
            axis=1,
        )
        y = df["apnoe"]

        index = np.random.permutation(len(X))
        self.X = X.iloc[index]
        self.y = y.iloc[index]

    def fit(self):
        inner_arch = [20, 30, 10]
        activ = "relu"
        out_activ = "sigmoid"
        loss = "binary_crossentropy"
        optim = "adam"
        epochs = 80
        batch_size = 5

        arch = [self.X.shape[1]] + inner_arch + [1]

        self.model = keras.Sequential(
            [layers.Input([arch[0]])]
            + [
                layers.Dense(units=arch[i], activation=activ)
                for i in range(1, len(arch) - 1)
            ]
            + [layers.Dense(units=arch[-1], activation=out_activ)]
        )
        self.model.compile(loss=loss, optimizer=optim, metrics=["recall"])
        self.model.fit(
            self.X,
            self.y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )
        self.model.save(f"apnoe_model.keras")

    def predict(self, pat) -> float:
        return round(
            self.model.predict(self.X.iloc[pat // 2].to_numpy().reshape((1, 13)))[0, 0],
            2,
        )
