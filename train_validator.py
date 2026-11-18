"""
Train a separate image validator for LeafLens.

Expected folder layout:
validator_dataset/
    supported_leaf/
        image1.jpg
        image2.jpg
        ...
    unsupported/
        image1.jpg
        image2.jpg
        ...

Class 0 (supported_leaf): leaf photos from plant categories supported by
your PlantVillage disease classifier.
Class 1 (unsupported): non-leaf photos AND leaves/plants outside the
categories supported by your disease classifier.

Use varied, correctly labelled examples. The validator is only as reliable
as the examples used to train and test it.
"""
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, Model

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="Folder containing supported_leaf/ and unsupported/")
parser.add_argument("--output", default="plant_validator.keras")
parser.add_argument("--epochs", type=int, default=12)
args = parser.parse_args()

dataset_dir = Path(args.dataset)
required = ["supported_leaf", "unsupported"]
missing = [name for name in required if not (dataset_dir / name).is_dir()]
if missing:
    raise SystemExit(
        "Missing class folder(s): " + ", ".join(missing) +
        ". Create both folders inside the dataset directory."
    )

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    class_names=required,
    labels="inferred",
    label_mode="binary",
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    class_names=required,
    labels="inferred",
    label_mode="binary",
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

print("Class order:", train_ds.class_names)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# This is a separate validator. It does not modify the user's hand-built
# MobileNetV2 disease classifier.
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False

inputs = layers.Input(shape=(224, 224, 3), name="validator_input")
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(1, activation="sigmoid", name="unsupported_score")(x)
validator = Model(inputs, outputs, name="plant_image_validator")

validator.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        args.output, monitor="val_loss", save_best_only=True
    ),
]

validator.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    callbacks=callbacks,
)

# Save the final best-weight model in the requested location.
validator.save(args.output)
print(f"Saved validator to: {args.output}")
print("Remember: evaluate it on separate, unseen images before relying on it.")
