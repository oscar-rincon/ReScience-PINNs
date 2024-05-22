"""
@author: Maziar Raissi
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pyDOE import lhs
import time
import tensorflow as tf
from tensorflow import keras
import scipy.io
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


np.random.seed(1234)
tf.random.set_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, u0, v0, tb, X_f, layers, lb, ub):
        
        X0 = np.concatenate((x0, np.zeros_like(x0)), axis=1)  # (x0, 0)
        X_lb = np.concatenate((np.zeros_like(tb) + lb[0], tb), axis=1)  # (lb[0], tb)
        X_ub = np.concatenate((np.zeros_like(tb) + ub[0], tb), axis=1)  # (ub[0], tb)
        
        self.lb = lb
        self.ub = ub
        
        self.x0 = X0[:, 0:1]
        self.t0 = X0[:, 1:2]

        self.x_lb = X_lb[:, 0:1]
        self.t_lb = X_lb[:, 1:2]

        self.x_ub = X_ub[:, 0:1]
        self.t_ub = X_ub[:, 1:2]

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        self.u0 = u0
        self.v0 = v0

        # Initialize NNs
        self.layers = layers
        self.model = self.initialize_NN(layers)

        # Loss and optimizer
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()

        self.loss_metric = tf.keras.metrics.Mean(name='loss')

    def initialize_NN(self, layers):
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(input_shape=(2,)))

        for width in layers[1:]:
            model.add(keras.layers.Dense(width, activation='tanh'))

        model.add(keras.layers.Dense(2))

        return model

    @tf.function
    def compute_loss(self, x0, t0, u0, v0, x_lb, t_lb, x_ub, t_ub, x_f, t_f):
        u0_pred, v0_pred = self.model(tf.concat([x0, t0], 1)), self.model(tf.concat([x0, t0], 1))
        u_lb_pred, v_lb_pred = self.model(tf.concat([x_lb, t_lb], 1)), self.model(tf.concat([x_lb, t_lb], 1))
        u_ub_pred, v_ub_pred = self.model(tf.concat([x_ub, t_ub], 1)), self.model(tf.concat([x_ub, t_ub], 1))
        f_u_pred, f_v_pred = self.net_f_uv(x_f, t_f)

        loss = (self.loss_object(u0, u0_pred) +
                self.loss_object(v0, v0_pred) +
                self.loss_object(u_lb_pred, u_ub_pred) +
                self.loss_object(v_lb_pred, v_ub_pred) +
                self.loss_object(u_lb_pred, u_ub_pred) +
                self.loss_object(v_lb_pred, v_ub_pred) +
                self.loss_object(f_u_pred, tf.zeros_like(f_u_pred)) +
                self.loss_object(f_v_pred, tf.zeros_like(f_v_pred)))

        return loss

    @tf.function
    def net_uv(self, x, t):
        X = tf.concat([x, t], 1)
        return self.model(X)[:, 0:1], self.model(X)[:, 1:2]

    @tf.function
    def net_f_uv(self, x, t):
        u, v = self.net_uv(x, t)

        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(tf.gradients(u, x)[0], x)[0]

        v_t = tf.gradients(v, t)[0]
        v_xx = tf.gradients(tf.gradients(v, x)[0], x)[0]

        f_u = u_t + 0.5 * v_xx + (u ** 2 + v ** 2) * v
        f_v = v_t - 0.5 * u_xx - (u ** 2 + v ** 2) * u

        return f_u, f_v

    @tf.function
    def train_step(self, x0, t0, u0, v0, x_lb, t_lb, x_ub, t_ub, x_f, t_f):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x0, t0, u0, v0, x_lb, t_lb, x_ub, t_ub, x_f, t_f)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.loss_metric(loss)

    def train(self, nIter):
        # Convert numpy arrays to TensorFlow tensors
        x0_tf = tf.constant(self.x0, dtype=tf.float32)
        t0_tf = tf.constant(self.t0, dtype=tf.float32)
        u0_tf = tf.constant(self.u0, dtype=tf.float32)
        v0_tf = tf.constant(self.v0, dtype=tf.float32)
        x_lb_tf = tf.constant(self.x_lb, dtype=tf.float32)
        t_lb_tf = tf.constant(self.t_lb, dtype=tf.float32)
        x_ub_tf = tf.constant(self.x_ub, dtype=tf.float32)
        t_ub_tf = tf.constant(self.t_ub, dtype=tf.float32)
        x_f_tf = tf.constant(self.x_f, dtype=tf.float32)
        t_f_tf = tf.constant(self.t_f, dtype=tf.float32)

        for epoch in range(nIter):
            self.train_step(x0_tf, t0_tf, u0_tf, v0_tf, x_lb_tf, t_lb_tf, x_ub_tf, t_ub_tf, x_f_tf, t_f_tf)

            if epoch % 100 == 0:
                print('Epoch: {}, Loss: {}'.format(epoch, self.loss_metric.result()))

    def predict(self, X_star):
        X_star_tf = tf.constant(X_star, dtype=tf.float32)
        u_star = self.model(X_star_tf)[:, 0:1].numpy()
        v_star = self.model(X_star_tf)[:, 1:2].numpy()
        f_u_star, f_v_star = self.net_f_uv(X_star_tf[:, 0:1], X_star_tf[:, 1:2])
        return u_star, v_star, f_u_star.numpy(), f_v_star.numpy()


if __name__ == "__main__":
    noise = 0.0

    # Domain bounds
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi / 2])

    N0 = 50
    N_b = 50
    N_f = 20000
    layers = [2, 100, 100, 100, 100, 2]

    data = scipy.io.loadmat('../Data/NLS.mat')

    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u ** 2 + Exact_v ** 2)

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact_u.T.flatten()[:, None]
    v_star = Exact_v.T.flatten()[:, None]
    h_star = Exact_h.T.flatten()[:, None]

    ###########################

    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x, :]
    u0 = Exact_u[idx_x, 0:1]
    v0 = Exact_v[idx_x, 0:1]

    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t, :]

    X_f = lb + (ub - lb) * lhs(2, N_f)

    model = PhysicsInformedNN(x0, u0, v0, tb, X_f, layers, lb, ub)

    start_time = time.time()
    model.train(50000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    u_pred, v_pred, f_u_pred, f_v_pred = model.predict(X_star)
    h_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_h = np.linalg.norm(h_star - h_pred, 2) / np.linalg.norm(h_star, 2)
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error h: %e' % (error_h))

    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    V_pred = griddata(X_star, v_pred.flatten(), (X, T), method='cubic')
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method='cubic')

    FU_pred = griddata(X_star, f_u_pred.flatten(), (X, T), method='cubic')
    FV_pred = griddata(X_star, f_v_pred.flatten(), (X, T), method='cubic')

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################

    X0 = np.concatenate((x0, 0 * x0), 1)  # (x0, 0)
    X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
    X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb, X_ub])

    fig, ax = newfig(1.0, 0.9)
    ax.axis('off')

    ####### Row 0: h(t,x) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 / 3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,
            clip_on=False)

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[75] * np.ones((2, 1)), line, 'k--', linewidth=1)
    ax.plot(t[100] * np.ones((2, 1)), line, 'k--', linewidth=1)
    ax.plot(t[125] * np.ones((2, 1)), line, 'k--', linewidth=1)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc='best')
    #    plt.setp(leg.get_texts(), color='w')
    ax.set_title('$|h(t,x)|$', fontsize=10)

    ####### Row 1: h(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, Exact_h[:, 75], 'b-', linewidth=2, label='Exact')
    ax.plot(x, H_pred[75, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.set_title('$t = %.2f$' % (t[75]), fontsize=10)
    ax.axis('square')
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, Exact_h[:, 100], 'b-', linewidth=2, label='Exact')
    ax.plot(x, H_pred[100, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])
    ax.set_title('$t = %.2f$' % (t[100]), fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, Exact_h[:, 125], 'b-', linewidth=2, label='Exact')
    ax.plot(x, H_pred[125, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])
    ax.set_title('$t = %.2f$' % (t[125]), fontsize=10)

    # savefig('./figures/NLS')
    plt.show()