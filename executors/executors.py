
import os
import json
from typing import Optional, Union

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import keras
import optuna

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
from utils.data_loader import apply_pca_on_X, cv_fold_split


os.environ["KERAS_BACKEND"] = "torch"
if torch.backends.mps.is_available():
    print("Apple's MPS backend (used by PyTorch on M1/M2 Macs) does not support float64 (double precision).")
    torch.set_default_dtype(torch.float32)
    mee = MeanEuclidianError("mee", dtype=torch.float32)
    print("set default to float32")
if keras.backend.backend() != "torch":
    print(f"warning: keras backend is set to {keras.backend.backend()}, restart jupyter kernel!!!!")
    raise RuntimeError()

UnitsType = Union[list[tuple[int, int]], list[int]]

class OptunaRegressorExecutor:
    """
    Optuna-based hyperparameter search executor for Keras regression models.

    Mirrors the business logic you posted:
      - (optional) PCA reduction on X
      - loop over units configs
      - for each units config: KFold CV
      - optimize study and save trials dataframe + best hp json

    Assumptions / required functions in your project:
      - build_model_single_hidden(...)
      - build_model_two_hidden(...)
      - apply_pca_on_X(train_dataset, pca_input_size, standardize=False) -> dataset-like object
      - cv_fold_split(dataset, train_idx, valid_idx, batch_size) -> (train_loader_cv, validation_loader_cv)
    """

    def __init__(
        self,
        train_loader,
        units: UnitsType,
        optuna_base_path: str = "keras/models/optuna",
        pca_input_size: Optional[int] = None,
        use_pca: bool = False,
        standardize_pca: bool = False,
        seed: int = 42,
        n_splits: int = 5,
        batch_size: int = 80,
        epochs: int = 1500,
        n_trials: int = 100,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        study_prefix: str = "keras-ML-CUP-",
        verbose: int = 0,
        baseline: float = None,
        n_jobs: int = 1,
    ):
        self.train_loader = train_loader
        self.units = units
        self.optuna_base_path = optuna_base_path

        self.use_pca = use_pca
        self.standardize_pca = standardize_pca
        self.pca_input_size = (
            pca_input_size
            if pca_input_size is not None
            else self.train_loader.dataset.X.shape[1]
        )

        self.seed = seed
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_trials = n_trials
        self.sampler = sampler
        self.study_prefix = study_prefix
        self.verbose = verbose
        self.baseline = baseline
        self.n_jobs = n_jobs

    # ----------------------------
    # Utilities
    # ----------------------------

    @staticmethod
    def _unit_name(u: Union[int, tuple[int, int]]) -> tuple[int, Optional[int], str]:
        if isinstance(u, int):
            unit1, unit2 = u, None
        elif isinstance(u, tuple):
            unit1, unit2 = u
        else:
            raise TypeError(f"Invalid unit spec: {u}")

        name = str(unit1)
        if unit2 is not None:
            name += "x" + str(unit2)
        return unit1, unit2, name

    def _ensure_dirs(self, base: str):
        os.makedirs(base, exist_ok=True)
        os.makedirs(os.path.join(base, "checkpoints"), exist_ok=True)

    # ----------------------------
    # Objective pieces
    # ----------------------------

    def _general_objective(
        self,
        trial: optuna.Trial,
        u: tuple[int, Optional[int]],
        path: str,
        train_loader_cv,
        validation_loader_cv,
    ) -> float:
        unit1, unit2 = u

        # Suggest hyperparameters (exactly as in your snippet)
        lambda_1 = trial.suggest_float("lambda_1", 3e-3, 1e-1, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True)
        activation_1 = trial.suggest_categorical("activation_1", ["relu", "gelu", "leaky_relu"])
        dropout_1 = trial.suggest_float("dropout_1", 0.2, 0.5, log=True)
        meta = {"n_outputs_": self.train_loader.dataset.y.shape[1], "n_features_in_": self.pca_input_size if self.use_pca else self.train_loader.dataset.X.shape[1]}

        # Build model
        if unit2 is not None:
            lambda_2 = trial.suggest_float("lambda_2", 3e-3, 1e-1, log=True)
            activation_2 = trial.suggest_categorical("activation_2", ["relu", "gelu", "leaky_relu"])
            dropout_2 = trial.suggest_float("dropout_2", 0.2, 0.5, log=True)

            model = build_model_two_hidden(
                meta=meta,
                unit1=unit1,
                unit2=unit2,
                learning_rate=learning_rate,
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                activation_1=activation_1,
                activation_2=activation_2,
                dropout_1=dropout_1,
                dropout_2=dropout_2,
                seed=self.seed,
            )
        else:
            model = build_model_single_hidden(
                meta=meta,
                unit1=unit1,
                learning_rate=learning_rate,
                lambda_1=lambda_1,
                activation_1=activation_1,
                dropout_1=dropout_1,
                seed=self.seed,
            )

        checkpoint_path = f"{path}/checkpoints/optuna_trial_{trial.number}.keras"
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
        )

        early_stopping_cb = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            baseline=self.baseline,
            patience=50,
            verbose=1,
            min_delta=1e-5,
            restore_best_weights=True,
        )

        # Train model
        history = model.fit(
            train_loader_cv,
            validation_data=validation_loader_cv,
            epochs=self.epochs,
            verbose=self.verbose,
            callbacks=[checkpoint_cb, early_stopping_cb],
        )

        val_loss = history.history["val_loss"][-1]
        return float(val_loss)

    # ----------------------------
    # Execution
    # ----------------------------

    def execute(self):
        # Prepare dataset (optionally PCA), matching your business logic.
        train_dataset = self.train_loader.dataset
        if self.use_pca:
            reduced_dataset = apply_pca_on_X(
                train_dataset,
                self.pca_input_size,
                standardize=self.standardize_pca,
            )
        else:
            reduced_dataset = train_dataset

        os.makedirs(self.optuna_base_path, exist_ok=True)

        for u in self.units:
            print(f"===unit {u}===")
            unit1, unit2, name = self._unit_name(u)
            u_tuple = (unit1, unit2)

            optuna_unitspecific_path = f"{self.optuna_base_path}/{name}"
            self._ensure_dirs(optuna_unitspecific_path)

            # CV objective
            def objective_cv(trial: optuna.Trial):
                fold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
                scores = []

                # IMPORTANT: same pattern you used: fold.split(range(len(dataset)))
                for fold_idx, (train_idx, valid_idx) in enumerate(
                    fold.split(range(len(reduced_dataset)))
                ):
                    print(f"[Trial {trial.number}] Executing fold {fold_idx}")

                    train_loader_cv, validation_loader_cv = cv_fold_split(
                        reduced_dataset,
                        train_idx,
                        valid_idx,
                        self.batch_size,
                    )

                    score = self._general_objective(
                        trial,
                        u_tuple,
                        optuna_unitspecific_path,
                        train_loader_cv,
                        validation_loader_cv,
                    )
                    scores.append(score)

                return float(np.mean(scores))

            study = optuna.create_study(
                study_name=self.study_prefix + name,
                sampler=self.sampler,
                direction="minimize",
            )

            study.optimize(objective_cv, n_trials=self.n_trials, n_jobs=self.n_jobs)

            df = study.trials_dataframe()
            optuna_results_path = f"{self.optuna_base_path}/{name}/optuna_results.csv"
            df.to_csv(optuna_results_path, index=False)

            # Your snippet writes hp.json to optuna_base_path (not unitspecific path).
            # Keeping that exact behavior.
            with open(f"{self.optuna_base_path }/{name}/hp.json", "w+") as f:
                json.dump(study.best_trial.params, f, indent=2)

            print(f"Saved: {optuna_results_path}")
            print(f"Saved: {self.optuna_base_path + '/hp.json'}")


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
            save_path="keras/models/rs",
            pipeline=None
            ):
        self.scoring = scoring
        self.n_iter = n_iter
        self.use_PCA = use_PCA
        self.train_loader = train_loader
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
            "pca__n_components": [2],
            "reg__model__seed": [self.seed],
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
                callbacks__0__patience=50,
                callbacks__0__verbose=1,
                callbacks__0__min_delta=1e-5,
                callbacks__0__restore_best_weights=True,
                loss="mean_squared_error",
                metrics=[mee]
            )

    def make_pipeline(self, regressor, n_components:int=None):
        if self.use_PCA:
            print("using PCA")
            return Pipeline([
                ("pca", PCA(n_components=n_components)),
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

            if not self.use_PCA:
                print("not using PCA, fixing param_distributions")
                param_distributions_copy.pop("pca__n_components", None)
                print(f"pca__n_components set to None")

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
            if self.use_PCA:
                saved_n_components = cleaned_hp["pca__n_components"]
                cleaned_hp.pop("pca__n_components", None)

            reg.set_params(**cleaned_hp)
            pipeline = self.make_pipeline(reg, n_components=saved_n_components)
            train_dataset = self.train_loader.dataset
            pipeline.fit(
                X=train_dataset.X,
                y=train_dataset.y,
            )
            reg.model_.save(save_path_subfolder+"/model.keras")
            ukeras.save_history_from_dict(reg.history_, save_path_subfolder)


def build_model_single_hidden(
    meta,
    unit1,
    seed,
    learning_rate,
    dropout_1,
    lambda_1,
    activation_1
):
    n_features = meta["n_features_in_"]
    output_size = meta["n_outputs_"]

    inputs = keras.Input(shape=(n_features,))
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
    meta,
    unit1, unit2,
    seed,
    learning_rate,
    dropout_1, dropout_2,
    lambda_1, lambda_2,
    activation_1, activation_2
):
    n_features = meta["n_features_in_"]
    output_size = meta["n_outputs_"]

    inputs = keras.Input(shape=(n_features,))
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