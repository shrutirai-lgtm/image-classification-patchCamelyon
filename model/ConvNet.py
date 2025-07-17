import tensorflow as tf

tf.keras.mixed_precision.set_global_policy("mixed_float16")
JIT_COMPILE = True  

class ConvNet:
    def __init__(self, 
                 input_shape, 
                 num_classes,
                 *,
                 optimizer='rmsprop', 
                 loss="sparse_categorical_crossentropy", 
                 metrics=["accuracy"], 
                 epochs=100, 
                 batch_size=32, 
                 verbose=1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = self._build_model()

    def _augment_data(self):
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.4),
            tf.keras.layers.RandomTranslation(0.2, 0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomBrightness(factor=0.4),
            tf.keras.layers.RandomContrast(factor=0.4)
        ])

    def _build_model(self):
        model = tf.keras.Sequential([
            self._augment_data(),
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax', dtype="float32")
        ])
        return model
    
    def _checkpoint(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath='best_model.keras',  
            monitor='val_accuracy',  
            mode='max',     
            save_best_only=True,    
            verbose=1
            )
    
    def compile_model(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics, jit_compile=JIT_COMPILE)

    def train_model(self, train_data, val_data, epochs=10, batch_size=32, verbose=1):
        self.model.fit(train_data, 
                       validation_data=val_data, 
                       epochs=epochs,
                       batch_size=batch_size, 
                       callbacks=[self._checkpoint()],
                       verbose=verbose
        )
        self.load_model("best_model.keras")

    def evaluate_model(self, test_data):
        return self.model.evaluate(test_data)
    
    def save_model(self, path): 
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def predict(self, x):
        return self.model.predict(x)
    
    def summary(self):
        self.model.summary()