import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot

def prune_model(model, final_sparsity):
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Define model for pruning.
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                final_sparsity=final_sparsity,
                                                                begin_step=0,
                                                                end_step=1000)
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    pruned_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    return pruned_model

def convert_into_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    return tflite_model

def quantise_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_tflite_model = converter.convert()

    return quantized_tflite_model

def evaluate_quantised_model_accuracy(original_model, quantized_tflite_model, test_images, test_labels):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=quantized_tflite_model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on some input data.
    input_shape = input_details[0]['shape']
    acc=0
    for i in range(len(test_images)):
        input_data = test_images[i].reshape(input_shape)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        if(np.argmax(output_data) == np.argmax(test_labels[i])):
            acc+=1
    acc = acc/len(test_images)
    print(acc*100)

def evaluate_accuarcy_pruned_model(original_model, pruned_model, test_images, test_labels):
    test_loss, test_accuracy = pruned_model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_accuracy)
