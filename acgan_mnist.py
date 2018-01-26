import os, sys
sys.path.append(os.getcwd())

import time
from random import randint

import matplotlib
matplotlib.use('Agg')
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.plot


DIM = 64  # Model dimensionality
CRITIC_ITERS = 5  # How many iterations to train the critic for
BATCH_SIZE = 64
ITERS = 200000
LAMBDA = 10  # Gradient penalty lambda hyperparameter
OUTPUT_DIM = 28*28  # Number of pixels in each iamge
CLASSES = 10  # Number of classes
PREITERATIONS = 1000  # Number of preiteration training cycles to run

lib.print_model_settings(locals().copy())


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)


def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)


def Generator(n_samples, numClasses, labels, bn=False, noise=None, condition=None):

    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    labels = tf.cast(labels, dtype=tf.float32)
    noise = tf.concat([noise, labels], axis=1)

    output = lib.ops.linear.Linear('Generator.Input', input_dim=128+numClasses, output_dim=4*4*4*DIM, inputs=noise)
    if bn:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', axes=[0], inputs=output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, shape=[-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', input_dim=4*DIM, output_dim=2*DIM, filter_size=5, inputs=output)
    if bn:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', axes=[0, 2, 3], inputs=output)
    output = tf.nn.relu(output)

    output = output[:, :, :7, :7]

    output = lib.ops.deconv2d.Deconv2D('Generator.3', input_dim=2*DIM, output_dim=DIM, filter_size=5, inputs=output)
    if bn:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', axes=[0, 2, 3], inputs=output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.Out', input_dim=DIM, output_dim=1, filter_size=5, inputs=output)
    output = tf.nn.sigmoid(output)

    return tf.reshape(output, shape=[-1, OUTPUT_DIM]), labels


def Discriminator(inputs, numClasses, bn=False):

    output = tf.reshape(inputs, shape=[-1, 28, 28, 1])

    output = lib.ops.conv2d.Conv2D('Discriminator.Input', input_dim=1, output_dim=DIM, filter_size=5, inputs=output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', input_dim=DIM, output_dim=2*DIM, filter_size=5, inputs=output, stride=2)
    if bn:
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', axes=[0, 2, 3], inputs=output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', input_dim=2*DIM, output_dim=4*DIM, filter_size=5, inputs=output, stride=2)
    if bn:
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', axes=[0, 2, 3], inputs=output)
    output = LeakyReLU(output)

    output = tf.reshape(output, shape=[-1, 4*4*4*DIM])

    sourceOutput = lib.ops.linear.Linear('Discriminator.sourceOutput', input_dim=4*4*4*DIM, output_dim=1, inputs=output)
    classOutput = lib.ops.linear.Linear('Discriminator.classOutput', input_dim=4*4*4*DIM, output_dim=numClasses, inputs=output)

    return tf.reshape(sourceOutput, shape=[-1]), tf.reshape(classOutput, shape=[-1, numClasses])


def genRandomLabels(n_samples, numClasses, condition=None):
    labels = np.zeros([n_samples, CLASSES], dtype=np.float32)
    for i in range(n_samples):
        if condition is not None:
            labelNum = condition
        else:
            labelNum = randint(0, numClasses - 1)
        labels[i, labelNum] = 1
    return labels


all_real_data = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
all_real_labels = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, CLASSES])

generated_labels = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, CLASSES])
samples_labels = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, CLASSES])

#gen_costs, disc_costs = [], []

fake_data, fake_labels = Generator(BATCH_SIZE, CLASSES, generated_labels)

# set up discriminator results
disc_fake, disc_fake_class = Discriminator(fake_data, CLASSES)
disc_real, disc_real_class = Discriminator(all_real_data, CLASSES)

prediction = tf.argmax(disc_fake_class, 1)
correct_answer = tf.argmax(fake_labels, 1)
equality = tf.equal(prediction, correct_answer)
genAccuracy = tf.reduce_mean(tf.cast(equality, dtype=tf.float32))

prediction = tf.argmax(disc_real_class, 1)
correct_answer = tf.argmax(all_real_labels, 1)
equality = tf.equal(prediction, correct_answer)
realAccuracy = tf.reduce_mean(tf.cast(equality, dtype=tf.float32))

gen_cost = -tf.reduce_mean(disc_fake)
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

gen_cost_test = -tf.reduce_mean(disc_fake)
disc_cost_test = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

generated_class_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_class, labels=fake_labels))
real_class_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_class, labels=all_real_labels))

gen_cost += generated_class_cost
disc_cost += real_class_cost

alpha = tf.random_uniform(shape=[BATCH_SIZE,1], minval=0., maxval=1.)
differences = fake_data - all_real_data
interpolates = all_real_data + (alpha*differences)
gradients = tf.gradients(Discriminator(interpolates, CLASSES)[0], [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)
disc_cost += LAMBDA*gradient_penalty

real_class_cost_gradient = real_class_cost*50 + LAMBDA*gradient_penalty

#gen_costs.append(gen_cost)
#disc_costs.append(disc_cost)

gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=lib.params_with_name('Generator'))
disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'))
class_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(real_class_cost_gradient, var_list=lib.params_with_name('Discriminator.'))

# for generating samples
fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
fixed_noise_samples = Generator(BATCH_SIZE, CLASSES, samples_labels, noise=fixed_noise)[0]


def generate_images(iteration):
    for j in range(CLASSES):
        curLabel = genRandomLabels(BATCH_SIZE, CLASSES, condition=j)
        samples = sess.run(fixed_noise_samples, feed_dict={samples_labels: curLabel})
        lib.save_images.save_images(samples.reshape((BATCH_SIZE, 28, 28)), './out/samples_{}_{}.png'.format(str(j), iteration))


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

with tf.Session() as sess:

    if not os.path.exists('./out/'):
        os.makedirs('./out/')

    sess.run(tf.global_variables_initializer())

    for iterp in range(PREITERATIONS):

        start_time = time.time()
        batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
        _, accuracy = sess.run([disc_train_op, realAccuracy],
                               feed_dict={all_real_data: batch_x, all_real_labels: batch_y, generated_labels: genRandomLabels(BATCH_SIZE, CLASSES)})
        if iterp % 100 == 99:
            print('Iter:{} Pretraining accuracy:{} Time taken:{}'.format(iterp, accuracy, time.time() - start_time))

    for it in range(ITERS):
        start_time = time.time()
        # Train generator
        if iter > 0:
            _ = sess.run(gen_train_op, feed_dict={generated_labels: genRandomLabels(BATCH_SIZE, CLASSES)})
        # Train critic
        for i in range(CRITIC_ITERS):
            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)

            _disc_cost, _disc_cost_test, _real_class_cost, _generated_class_cost, _gen_cost_test, _genAccuracy, _realAccuracy, _ = \
                sess.run([disc_cost, disc_cost_test, real_class_cost, generated_class_cost, gen_cost_test, genAccuracy, realAccuracy, disc_train_op],
                    feed_dict={all_real_data: batch_x, all_real_labels: batch_y, generated_labels: genRandomLabels(BATCH_SIZE, CLASSES)})

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)
        lib.plot.plot('train disc test cost', _disc_cost_test)
        lib.plot.plot('real class cost', _real_class_cost)
        lib.plot.plot('generated class cost', _generated_class_cost)
        lib.plot.plot('generated test cost', _gen_cost_test)
        lib.plot.plot('gen accuracy', _genAccuracy)
        lib.plot.plot('real accuracy', _realAccuracy)

        if it % 100 == 99:
            generate_images(iteration=it)

        if (it < 10) or (it % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()