import tensorflow as tf


class CFG:
    PREPROCESS = False
    PRETRAINED = True
    EPOCHS = 20
    BATCH_SIZE = 4096
    LR = 1e-3
    WD = 0.05
    # Number of folds
    NBR_FOLDS = 15

    # Only the first fold selected
    SELECTED_FOLDS = [0, 1, 2, 3, 4]
    EXIST_MODELS = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
    ]

    SEED = 2222


def OneDCNN_model(strategy):
    with strategy.scope():
        INP_LEN = 142
        NUM_FILTERS = 32
        hidden_dim = 128

        inputs = tf.keras.layers.Input(shape=(INP_LEN,), dtype="int32")
        x = tf.keras.layers.Embedding(
            input_dim=36, output_dim=hidden_dim, input_length=INP_LEN, mask_zero=True
        )(inputs)
        x = tf.keras.layers.Conv1D(
            filters=NUM_FILTERS,
            kernel_size=3,
            activation="relu",
            padding="valid",
            strides=1,
        )(x)
        x = tf.keras.layers.Conv1D(
            filters=NUM_FILTERS * 2,
            kernel_size=3,
            activation="relu",
            padding="valid",
            strides=1,
        )(x)
        x = tf.keras.layers.Conv1D(
            filters=NUM_FILTERS * 3,
            kernel_size=3,
            activation="relu",
            padding="valid",
            strides=1,
        )(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)

        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        outputs = tf.keras.layers.Dense(3, activation="sigmoid")(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=CFG.LR, weight_decay=CFG.WD)
        loss = "binary_crossentropy"
        weighted_metrics = [tf.keras.metrics.AUC(curve="PR", name="avg_precision")]
        model.compile(
            loss=loss,
            optimizer=optimizer,
            weighted_metrics=weighted_metrics,
        )
        return model
