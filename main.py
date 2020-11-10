import os
import numpy as np
from PIL import Image
import tensorflow as tf
from core.nn import StyleContentModel

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

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

@tf.function
def train_step(image, extractor, style_targets, content_targets, optimizer):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = extractor.style_content_loss(outputs, style_targets, content_targets)
        # Add total variation loss
        tv_weight = 20.0
        loss += tv_weight * tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

def main():

    content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
    style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')


    content_image = load_image(content_path)
    style_image = load_image(style_path)

    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]
    content_layers = ["block4_conv2"]

    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)["style"]
    content_targets = extractor(content_image)["content"]

    os.makedirs("intermediate", exist_ok=True)

    image = tf.Variable(content_image)
    optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    for epoch in range(15):
        for i in range(100):
            train_step(image, extractor, style_targets, content_targets, optimizer)

        path = os.path.join("intermediate", f"{epoch}.png")
        tensor_to_image(image).save(path)


if __name__ == '__main__':
    main()
