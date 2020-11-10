import os
import numpy as np
from PIL import Image
import tensorflow as tf
from core.nn import StyleContentModel

# Select layers from VGG19 for style extraction
style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
# Select layer from VGG19 for content extraction
content_layers = ["block5_conv2"]

# Weights for loss function
style_weight = 1e-2
content_weight = 1e4
tv_weight = 20.0

def load_image(img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Get height and width of image and cast to float32
    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    # Limit maximum dimension to 512 pixels
    max_dim = 512
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    # Resize image
    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :]
    return image

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def style_content_loss(outputs, style_targets, content_targets):
    style_outputs = outputs["style"]
    content_outputs = outputs["content"]

    style_loss = [tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()]
    style_loss = tf.add_n(style_loss)
    style_loss *= style_weight / len(style_layers)

    content_loss = [tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()]
    content_loss = tf.add_n(content_loss)
    content_loss *= content_weight / len(content_layers)

    return style_loss + content_loss

@tf.function
def train_step(image, extractor, style_targets, content_targets, optimizer):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, style_targets, content_targets)
        # Add total variation loss
        loss += tv_weight * tf.image.total_variation(image)

    # Apply gradient
    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    # Clip image values
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))

def main():

    content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
    style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

    content_image = load_image(content_path)
    style_image = load_image(style_path)

    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)["style"]
    content_targets = extractor(content_image)["content"]

    image = tf.Variable(content_image)
    optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    os.makedirs("intermediate", exist_ok=True)
    for epoch in range(15):
        for i in range(100):
            # Train one step
            train_step(image, extractor, style_targets, content_targets, optimizer)
        # Save intermediate results
        path = os.path.join("intermediate", f"{epoch}.png")
        tensor_to_image(image).save(path)

if __name__ == '__main__':
    main()
