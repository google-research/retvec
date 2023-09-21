import tensorflow as tf

def clone_initializer(initializer):
    # Keras initializer is going to be stateless, which mean reusing the same
    # initializer will produce same init value when the shapes are the same.
    if isinstance(initializer, tf.keras.initializers.Initializer):
    return initializer.__class__.from_config(initializer.get_config())
    # When the input is string/dict or other serialized configs, caller will
    # create a new keras Initializer instance based on that, and we don't need to
    # do anything
    return initializer
