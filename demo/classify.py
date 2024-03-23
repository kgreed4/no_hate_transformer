import tensorflow as tf

def classify_text(loaded_model, tokenizer, text, maximum_position_encoding=128):
    """
    Classifies input text using a pre-trained model and tokenizer.

    Parameters:
    - loaded_model: The loaded TensorFlow SavedModel for text classification.
    - tokenizer: The tokenizer corresponding to the pre-trained model.
    - text: The input text to classify.
    - maximum_position_encoding: The maximum length of input texts that the model can handle.

    Returns:
    - predicted_class: The class predicted by the model for the input text.
    """
    # Tokenizing the input text
    inputs = tokenizer.encode_plus(
        text,
        max_length=maximum_position_encoding,
        truncation=True,
        padding="max_length",
        return_tensors="tf"
    )

    # Adjust input names according to the model's expected signature
    model_inputs = {'input_1': inputs['input_ids'], 'input_2': inputs['attention_mask']}

    # Making prediction using the serving_default signature
    serving_output = loaded_model.signatures['serving_default'](**model_inputs)

    # Assuming the output from the model is logits and named 'output_1'
    logits = serving_output['output_1']

    # Converting logits to probabilities
    probabilities = tf.nn.softmax(logits, axis=-1)

    # Extracting the predicted class
    predicted_class = tf.argmax(probabilities, axis=-1).numpy()[0]

    return predicted_class
