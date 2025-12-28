from keras import Loss, ops
import keras

@keras.saving.register_keras_serializable(package="losses")
class MeanEuclidianError(Loss):
    def __init__(self, name=None, reduction="sum_over_batch_size", dtype=None):
        super().__init__(name, reduction, dtype)

    def call(self, y_true, y_pred):
            return ops.mean(ops.sqrt(ops.sum(ops.square(y_pred - y_true), axis=-1)))
    
    def get_config(self):
        return {"name": self.name, "reduction": self.reduction, "dtype": self.dtype}