from torch.utils.data import Dataset, DataLoader
import keras
import torch
import os
import json
import pandas as pd
import utils.keras as ukeras
from scikeras.wrappers import KerasRegressor, BaseWrapper
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.stats import loguniform
from losses import MeanEuclidianError

os.environ["KERAS_BACKEND"] = "torch"
if torch.backends.mps.is_available():
    print("Apple's MPS backend (used by PyTorch on M1/M2 Macs) does not support float64 (double precision).")
    torch.set_default_dtype(torch.float32)
    mee = MeanEuclidianError("mee", dtype=torch.float32)
    print("set default to float32")
if keras.backend.backend() != "torch":
    print(f"warning: keras backend is set to {keras.backend.backend()}, restart jupyter kernel!!!!")
    raise RuntimeError()

class RandomizedSearchRegressionExecutor:
    def __init__(
            self,
            train_loader: DataLoader,
            loss: str,
            scoring: str,
            units: list[tuple[int]] | list[int],
            epochs=1500,
            batch_size=80,
            verbose=0,
            validation_split=.20,
            baseline: float=None,
            seed=42,
            param_distributions: dict[str,]=None,
            use_PCA=False,
            n_iter = 100,
            pca_input_size: int=None,
            save_path="keras/models/rs",
            pipeline=None
            ):
        self.scoring = scoring
        self.n_iter = n_iter
        self.use_PCA = use_PCA
        self.train_loader = train_loader
        self.pca_input_size = pca_input_size if pca_input_size is not None else self.train_loader.dataset.X.shape[1]
        self.save_path = save_path
        self.loss = loss
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.validation_split = validation_split
        self.baseline = baseline
        self.seed = seed
        self.pipeline = pipeline
        if param_distributions is None:
            self.params_init()
        else: 
            self.param_distributions = param_distributions

    def params_init(self):
        param_distributions = {
            "reg__model__learning_rate": loguniform(1e-3, 1e-2),
            "reg__model__lambda_1": loguniform(3e-3, 1e-1),
            "reg__model__lambda_2": loguniform(3e-3, 1e-1),
            "reg__model__activation_1": ["relu", "gelu", "leaky_relu"],
            "reg__model__activation_2": ["relu", "gelu", "leaky_relu"],
            "reg__model__dropout_1": loguniform(0.2, 0.5),
            "reg__model__dropout_2": loguniform(0.2, 0.5),
            "reg__model__pca_input_size": [self.pca_input_size],
            "reg__model__seed": [self.seed],
            "reg__model__output_size": [self.train_loader.dataset.y.shape[1]],
        }
        print("using default param_distributions", param_distributions)
        self.param_distributions = param_distributions

    def keras_regressor(self, unit2):
        build_fn = build_model_two_hidden if unit2 is not None else build_model_single_hidden
        return KerasRegressor(
                model=build_fn,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=self.verbose,
                validation_split=self.validation_split,
                validation_batch_size=80,
                callbacks=[keras.callbacks.EarlyStopping],
                callbacks__0__monitor="val_loss",
                callbacks__0__baseline=self.baseline,
                callbacks__0__patience=20,
                callbacks__0__verbose=1,
                callbacks__0__restore_best_weights=True,
                loss="mean_squared_error",
                metrics=[mee]
            )

    def make_pipeline(self, regressor):
        if self.use_PCA:
            print("using PCA")
            return Pipeline([
                ("pca", PCA(n_components=self.pca_input_size)),
                ("reg", regressor)
            ])
        
        print("not using PCA")
        return Pipeline([
                ("reg", regressor)
            ])

    def execute(self):
        for u in self.units:
            print(f"===unit {u}===")
            
            ## randomized search
            param_distributions_copy = dict(self.param_distributions)
            if type(u) is int:
                unit1, unit2 = u, None
                param_distributions_copy.pop("reg__model__dropout_2", None)
                param_distributions_copy.pop("reg__model__activation_2", None)
                param_distributions_copy.pop("reg__model__lambda_2", None)
            
            if type(u) is tuple:
                unit1, unit2 = u
                param_distributions_copy["reg__model__unit2"] = [unit2]

            param_distributions_copy["reg__model__unit1"] = [unit1]
            name = str(unit1)
            if unit2 is not None:
                name += "x" + str(unit2)
            save_path_subfolder = f"{self.save_path}/{name}"
            k = 5
            cv = KFold(n_splits=k, shuffle=True, random_state=self.seed)

            reg = self.keras_regressor(unit2)
            pipeline = self.make_pipeline(reg)
            self.pipeline = pipeline
            print(param_distributions_copy)
            random_search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_distributions_copy,
                n_iter=self.n_iter,
                cv=cv,
                scoring=self.scoring,
                verbose=0,
                random_state=self.seed,
            )
            random_search.fit(self.train_loader.dataset.X, self.train_loader.dataset.y)
            rs_hp = random_search.best_params_
            print("Best CV MEE (negative):", random_search.best_score_)
            os.makedirs(save_path_subfolder, exist_ok=True)
            with open(save_path_subfolder + "/hp.json", "w") as f:
                json.dump(random_search.best_params_, f, indent=2)
                print("random_search hp saved")

            results_df = pd.DataFrame(random_search.cv_results_)
            results_df = results_df.rename(columns=lambda x: x.replace('param_', 'params_'))
            results_df.to_csv(f"{save_path_subfolder}/cv_results_df.csv", index=False)
            cleaned_hp = {k.replace('reg__', ''): v for k, v in rs_hp.items()}
            reg.set_params(**cleaned_hp)
            pipeline = self.make_pipeline(reg)
            train_dataset = self.train_loader.dataset
            pipeline.fit(
                X=train_dataset.X,
                y=train_dataset.y,
            )
            reg.model_.save(save_path_subfolder+"/model.keras")
            ukeras.save_history_from_dict(reg.history_, save_path_subfolder)


def build_model_single_hidden(
    unit1,
    pca_input_size,
    seed,
    output_size,
    learning_rate,
    dropout_1,
    lambda_1,
    activation_1
):
    inputs = keras.Input(shape=(pca_input_size,))
    x = inputs

    x = keras.layers.Dense(
            unit1,
            activation=activation_1,
            kernel_regularizer=keras.regularizers.l2(lambda_1),
            kernel_initializer=keras.initializers.GlorotNormal(seed=seed)
        )(x)
    
    x = keras.layers.Dropout(
        rate=dropout_1
    )(x)
    
    # Output layer
    outputs = keras.layers.Dense(output_size)(x)

    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=0.9,
        nesterov=True,
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[mee],
    )

    return model

def build_model_two_hidden(
    unit1, unit2,
    pca_input_size,
    seed,
    output_size,
    learning_rate,
    dropout_1, dropout_2,
    lambda_1, lambda_2,
    activation_1, activation_2
):
    inputs = keras.Input(shape=(pca_input_size,))
    x = inputs

    x = keras.layers.Dense(
            unit1,
            activation=activation_1,
            kernel_regularizer=keras.regularizers.l2(lambda_1),
            kernel_initializer=keras.initializers.GlorotNormal(seed=seed)
        )(x)
    
    x = keras.layers.Dropout(
        rate=dropout_1
    )(x)
    
    x = keras.layers.Dense(
            unit2,
            activation=activation_2,
            kernel_regularizer=keras.regularizers.l2(lambda_2),
            kernel_initializer=keras.initializers.GlorotNormal(seed=seed+1)
        )(x)

    x = keras.layers.Dropout(
        rate=dropout_2
    )(x)

    # Output layer
    outputs = keras.layers.Dense(output_size)(x)

    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=0.9,
        nesterov=True,
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[mee],
    )

    return model