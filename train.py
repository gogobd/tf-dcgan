#!/usr/bin/env python

import tensorflow
from dcgan import DCGAN

FLAGS = tensorflow.app.flags.FLAGS

tensorflow.app.flags.DEFINE_integer(
    'max_steps', 10001, """Number of batches to run.""")
tensorflow.app.flags.DEFINE_string(
    'data_dir', 'data', """Path to the TFRecord data directory.""")

# tensorflow.app.flags.DEFINE_string(
#     'images_dir', 'images', """Directory where to write generated images.""")

# tensorflow.app.flags.DEFINE_string('logdir', 'logdir',
#     """Directory where to write event logs and checkpoint.""")
# tensorflow.app.flags.DEFINE_integer('num_examples_per_epoch_for_train', 5000,
#     """number of examples for train""")


def get_images_batch():
    return []


dcgan = DCGAN(
    g_depths=[8192, 4096, 2048, 1024, 512, 256, 128],
    d_depths=[64, 128, 256, 512, 1024, 2048, 4096]
)
train_images = get_images_batch()
losses = dcgan.loss(train_images)
train_op = dcgan.train(losses)

with tensorflow.Session() as sess:
    sess.run(tensorflow.global_variables_initializer())

    for step in range(FLAGS.max_steps):
        _, g_loss_value, d_loss_value = sess.run(
            [train_op, losses[dcgan.g], losses[dcgan.d]]
        )


images = dcgan.sample_images()
with tensorflow.Session() as sess:
    # restore trained variables

    generated = sess.run(images)
    with open('output.binary', 'wb') as f:
        f.write(generated)
