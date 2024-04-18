model = Sequential([
    Dense (units=16, input_shape=(1,), activation='relu'),
    Dense (units=32, activation='relu'),
    Dense (units=2, activation='softmax')
])

model.compile(optimizer= Adam(learning_rate=0.0001), loss ='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit( x=scaled_train_samples,
          y=train_labels,
          validation_split=0.1,
          batch_size=10,
          epochs=30,
          verbose=2)


model.save('models/medical_trial_model.h5')