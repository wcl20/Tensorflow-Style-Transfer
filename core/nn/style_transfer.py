import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model


class StyleContentModel(Model):

    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()

        self.vgg = self.vgg_layers(style_layers + content_layers)
        self.vgg.trainable = False

        self.style_layers = style_layers
        self.content_layers = content_layers

        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)

    def call(self, inputs):
        inputs = inputs * 255.
        preprocessed_input = preprocess_input(inputs)

        outputs = self.vgg(preprocessed_input)
        style_outputs = outputs[:self.num_style_layers]
        content_outputs = outputs[self.num_style_layers:]

        # Compute Gramm matrix for each style output
        style_outputs = [self.gram_matrix(style_output) for style_output in style_outputs]

        content_dict = { name: value for name, value in zip(self.content_layers, content_outputs) }
        style_dict = { name: value for name, value in zip(self.style_layers, style_outputs) }

        return { "content": content_dict, "style": style_dict }

    @staticmethod
    def vgg_layers(layer_names):
        vgg = VGG19(include_top=False, weights="imagenet")
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in layer_names]
        return Model([vgg.input], outputs)

    @staticmethod
    def gram_matrix(input_tensor):
        # Gram matrix is the dot product of input and its transpose
        result = tf.linalg.einsum("bijc, bijd->bcd", input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations

    @staticmethod
    def style_content_loss(outputs, style_targets, content_targets):
        style_outputs = outputs["style"]
        content_outputs = outputs["content"]

        style_weight = 1e-2
        style_loss = [tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()]
        style_loss = tf.add_n(style_loss)
        style_loss *= style_weight / len(style_outputs)

        content_weight = 1e4
        content_loss = [tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()]
        content_loss = tf.add_n(content_loss)
        content_loss *= content_weight / len(content_outputs)

        return style_loss + content_loss
