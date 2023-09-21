import re
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops


@tf.keras.utils.register_keras_serializable(package="tensorflow_retvec")
class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applys a warmup schedule on a given learning rate decay schedule."""

    def __init__(self, initial_learning_rate, decay_schedule_fn, warmup_steps, power=1.0, name=None):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(warmup_percent_done, self.power)
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step - self.warmup_steps),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }


def create_optimizer(init_lr, num_train_steps, num_warmup_steps, weight_decay_rate=0.0,
                     optimizer_type="adam", skip_adaptive=False,
                     beta_1=0.9, beta_2=0.999, end_lr=0.0, decay_fn='cosine'):
    """Creates an optimizer with learning rate schedule and optional weight decay and warmup."""
    if decay_fn == "linear":
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=init_lr, decay_steps=num_train_steps - num_warmup_steps, end_learning_rate=end_lr, power=1.0
        )
    elif decay_fn == "cosine":
        learning_rate_fn = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=init_lr, decay_steps=num_train_steps - num_warmup_steps, alpha=end_lr/init_lr
        )
    else:
        learning_rate_fn = init_lr

    if num_warmup_steps:
        learning_rate_fn = WarmUp(
            initial_learning_rate=init_lr, decay_schedule_fn=learning_rate_fn, warmup_steps=num_warmup_steps
        )
    layer_decay = None

    if optimizer_type == "adam":
        optimizer = tfa.optimizers.AdamW(
            weight_decay=weight_decay_rate,
            learning_rate=learning_rate_fn,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=1e-6,
            exclude_from_weight_decay=["layer_norm", "bias", "LayerNorm"],
        )
    elif optimizer_type == "lamb":
        if skip_adaptive:
            skip_list = ["layer_norm", "bias", "LayerNorm"]
        else:
            skip_list = ["None"]

        optimizer = tfa.optimizers.LAMB(
            learning_rate=learning_rate_fn,
            weight_decay_rate=weight_decay_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=1e-6,
            exclude_from_weight_decay=["layer_norm", "bias", "LayerNorm"],
            exclude_from_layer_adaptation=skip_list,
        )
    else:
        raise ValueError(f"Unsupported optimizer type {optimizer_type}")

    return optimizer
