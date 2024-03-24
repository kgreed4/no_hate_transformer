import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout, LayerNormalization, Layer, MultiHeadAttention
from tensorflow.keras.models import Model

# Here we create a custom PositionalEncoding layer
class PositionalEncoding(Layer):
    # The init will create a positional encoding layer. The sequence_length is the length of the input sequence 
    # and the d_model is the dimensionality of the model embeddings 
    def __init__(self, sequence_length, d_model, **kwargs):
        """
        Purpose: This function will create a positional encoding layer that will be used to add positional information to the input sequence.
        sequence_length: The length of the input sequence
        d_model: The dimensionality of the model embeddings (size of the embeddings into which input tokens are transformed)
        """
        super(PositionalEncoding, self).__init__(**kwargs)
        # The positional encoding tensor is created using the positional_encoding function
        # It is then stored within the self.pos_encoding variable
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)

    def get_angles(self, pos, i, d_model):
        """
        Purpose: This function will calculate the angles for the positional encoding. The formula is key 
        to making sure that each dimension of the positional encoding is unique and corresponds to a sinusoid of different wavelength.
        Basically, each word has a different value on the sinusoidal wave that makes it unique

        Here we calculate the angle values for each position in the input sequence and for each dimension in the model's embedding space

        pos: The position in the sequence
        i: The dimension of the model
        d_model: The dimensionality of the model
        """
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angles

    def positional_encoding(self, position, d_model):
        """
        Purpose: This function will be used to create the positional encoding matrix. 
        The function uses the get_angles function to calculate the angles for the positional encoding.
        The sine function is used for the even indices of the encoding and the cosine function is used for the odd indices.

        position: The position in the sequence
        d_model: The dimensionality of the model
        """
        angle_rads = self.get_angles(pos=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                                     i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                                     d_model=d_model)
        
        # By alternating between the sine and cosine functions, we can create the positional encoding matrix that is unique for each position
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        # Here we concatenate the sines and cosines to create the positional encoding matrix
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# Here we create a custom layer for Normalization
class AddNormalization(Layer):
    def __init__(self, **kwargs):
        """
        Purpose: This function will create a normalization layer that will be used to normalize the output of a sublayer.
        The layer normalization layer introduces the concept of skip connections (residual connections), which are used to add the output of a sublayer 
        """
        super(AddNormalization, self).__init__(**kwargs)
        # The variable self.layer_nrom is used to store the LayerNormalization layer
        self.layer_norm = LayerNormalization()
    
    def call(self, x, sublayer_x):
        return self.layer_norm(x + sublayer_x)

# Here we create the FeedForward layer
class FeedForward(Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        """
        Purpose: This function will create a feedforward layer that will consist of two dense layers.
        d_ff: The dimensionality of the feedforward layer
        d_model: The dimensionality of the model
        """
        super(FeedForward, self).__init__(**kwargs)
        # The layer will consist of two dense layers
        # The first dense layer will have a ReLU activation function
        self.dense1 = Dense(d_ff, activation='relu')
        self.dense2 = Dense(d_model)
    
    def call(self, x):
        return self.dense2(self.dense1(x))

# Here we create the MultiHeadAttention layer
class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, d_ff, rate, **kwargs):
        """
        Purpose: This function will create an encoder layer that will consist of a multi-head attention layer and a feedforward layer.
        Here we take the previous classes and combine them to create the encoder layer.
        d_model: The dimensionality of the model
        num_heads: The number of attention heads
        d_ff: The dimensionality of the feedforward layer
        rate: The dropout rate
        """
        super(EncoderLayer, self).__init__(**kwargs)
        # Here we create the multi-head attention layer
        self.mha = MultiHeadAttention(key_dim=d_model, num_heads=num_heads)
        self.ffn = FeedForward(d_ff, d_model)
        
        # Here we normalize the output of the multi-head attention layer
        self.add_norm1 = AddNormalization()  
        self.add_norm2 = AddNormalization() 

        # We include dropout layers to prevent overfitting
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training):
        """
        Purpose: This function will be used to apply the multi-head attention and feedforward layers to the input tensor.
        x: The input tensor
        training: Whether the model is in training mode
        """
        # Here we apply the multi-head attention layer to the input tensor
        attn_output = self.mha(x, x, x) 
        # Here we include dropout to the output of the multi-head attention layer 
        attn_output = self.dropout1(attn_output, training=training)
        # Here we normalize the output of the multi-head attention layer
        out1 = self.add_norm1(x, attn_output)
        # Here we apply the feedforward layer to the output of the multi-head attention layer
        ffn_output = self.ffn(out1)
        # Here we include dropout to the output of the feedforward layer
        ffn_output = self.dropout2(ffn_output, training=training)
        # Here we normalize the output of the feedforward layer
        return self.add_norm2(out1, ffn_output)

# Here we create the TransformerEncoder layer. This layer serves as the core of the Transformer encoder architecture,
# combining the embedding layer, custom PositionalEncoding layer, and multiple instances of the EncoderLayer (which incorporates multi-head
# attention and feedforward neural network layers). 
    
# Additionally, dropout is applied for regularization. This organized assembly allows for comprehensive processing of input sequences, 
# embedding them in a high-dimensional space, enriching them with positional information, and transforming them through a series of 
# attention and feedforward operations to capture complex patterns and dependencies within the data.

class TransformerEncoder(Model):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, maximum_position_encoding, rate=0.1):
        """
        Purpose: This function will create the structure of the full Transformer encoder model. Here we combine the previous classes
        to create the full encoder model. Such classes are the Embedding, PositionalEncoding, and EncoderLayer classes.
        num_layers: The number of encoder layers
        d_model: The dimensionality of the model
        num_heads: The number of attention heads
        d_ff: The dimensionality of the feedforward layer
        input_vocab_size: The size of the input vocabulary
        maximum_position_encoding: The maximum position encoding
        rate: The dropout rate
        """
        super(TransformerEncoder, self).__init__()
        # Here we create the d_model variable, which is the dimensionality of the model
        self.d_model = d_model
        # Here we introduce the number of encoder layers
        self.num_layers = num_layers
        # Here we create the embeddings based on the Embedding class
        self.embedding = Embedding(input_vocab_size, d_model)
        # Here we can create the positional encoding layer using the PositionalEncoding class
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, d_model)
        # Here we use the EncoderLayer class to create the encoder layers
        self.enc_layers = [EncoderLayer(d_model, num_heads, d_ff, rate) 
                           for _ in range(num_layers)]
        
        self.dropout = Dropout(rate)

    def call(self, x, training):
        seq_len = tf.shape(x)[1]
        # Here we create the input tensor using the embedding layer
        x = self.embedding(x) 
        # Here we scale the embedding by multiplying it by the square root of the model dimension
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # Here we add the positional encoding to the input tensor
        x += self.pos_encoding(x)
        # Here we include dropout to the input tensor
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training)
        
        return x 

# Here we create the TransformerClassifier model. This model is a simple classifier that uses the TransformerEncoder as its base.
class TransformerClassifier(Model):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, maximum_position_encoding, rate, num_classes):
        """
        Purpose: This function will create the structure of the full Transformer classifier model. Here we combine the previous classes
        and also add a softmax layer to create the classiying component of the model. 
        """
        super(TransformerClassifier, self).__init__()
        # The architecture will consist of the TransformerEncoder and a GlobalAveragePooling1D layer and a final Dense layer, which
        # includes the softmax activation function
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, maximum_position_encoding, rate)
        self.global_pool = GlobalAveragePooling1D()
        self.final_layer = Dense(num_classes, activation='softmax')
    
    # The call function will be used to apply the TransformerEncoder to the input tensor
    def call(self, inputs, training=False):
        input_ids, attention_mask = inputs
        # We apply the TransformerEncoder to the input tensor
        encoder_output = self.encoder(input_ids, training=training)
        # We apply the GlobalAveragePooling1D layer to the output of the TransformerEncoder
        pooled_output = self.global_pool(encoder_output)
        # We apply the final Dense layer, which includes the softmax activation function, to the output of the GlobalAveragePooling1D layer
        return self.final_layer(pooled_output)