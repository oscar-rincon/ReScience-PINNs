# Configuración del entorno y carga de módulos
import os
import sys
import time
import scipy.optimize
import scipy.io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pyDOE import lhs  # Latin Hypercube Sampling
#from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Configuración del entorno para usar CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, '../../Utilities/')

# Configuración de semillas para reproducibilidad
np.random.seed(1234)
tf.random.set_seed(1234)

# Definición de la clase del modelo secuencial
class Sequentialmodel(tf.Module): 
    def __init__(self, layers, name=None):
        self.W = []  # Pesos y sesgos
        self.parameters = 0  # Número total de parámetros
        self.iteration = 0
        for i in range(len(layers)-1):
            input_dim = layers[i]
            output_dim = layers[i+1]
            std_dv = np.sqrt((2.0/(input_dim + output_dim)))  # Desviación estándar de Xavier
            w = tf.random.normal([input_dim, output_dim], dtype='float64') * std_dv
            w = tf.Variable(w, trainable=True, name='w' + str(i+1))
            b = tf.Variable(tf.cast(tf.zeros([output_dim]), dtype='float64'), trainable=True, name='b' + str(i+1))
            self.W.append(w)
            self.W.append(b)
            self.parameters += input_dim * output_dim + output_dim
    
    def evaluate(self, x):
        x = 2*(x - lb) / (ub - lb) - 1
        a = x
        for i in range(len(layers) - 2):
            z = tf.add(tf.matmul(a, self.W[2*i]), self.W[2*i+1])
            a = tf.nn.tanh(z)
        a = tf.add(tf.matmul(a, self.W[-2]), self.W[-1])  # Para regresión, sin activación en la última capa
        return a
    
    def get_weights(self):
        parameters_1d = []
        for i in range(len(layers) - 1):
            w_1d = tf.reshape(self.W[2*i], [-1])  # Aplanar pesos
            b_1d = tf.reshape(self.W[2*i+1], [-1])  # Aplanar sesgos
            parameters_1d = tf.concat([parameters_1d, w_1d], 0)  # Concatenar pesos
            parameters_1d = tf.concat([parameters_1d, b_1d], 0)  # Concatenar sesgos
        return parameters_1d
    
    def set_weights(self, parameters):
        for i in range(len(layers) - 1):
            shape_w = tf.shape(self.W[2*i]).numpy()
            size_w = tf.size(self.W[2*i]).numpy()
            shape_b = tf.shape(self.W[2*i+1]).numpy()
            size_b = tf.size(self.W[2*i+1]).numpy()
            pick_w = parameters[0:size_w]
            self.W[2*i].assign(tf.reshape(pick_w, shape_w))
            parameters = np.delete(parameters, np.arange(size_w), 0)
            pick_b = parameters[0:size_b]
            self.W[2*i+1].assign(tf.reshape(pick_b, shape_b))
            parameters = np.delete(parameters, np.arange(size_b), 0)

    def optimizerfunc(self, parameters):
        self.set_weights(parameters)
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            loss_val = self.loss(x0_tf, x1_tf, dt, IRK_weights, u0_tf)
        grads = tape.gradient(loss_val, self.trainable_variables)
        grads_1d = []
        for i in range(len(layers) - 1):
            grads_w_1d = tf.reshape(grads[2*i], [-1])  # Aplanar pesos
            grads_b_1d = tf.reshape(grads[2*i+1], [-1])  # Aplanar sesgos
            grads_1d = tf.concat([grads_1d, grads_w_1d], 0)  # Concatenar gradientes de pesos
            grads_1d = tf.concat([grads_1d, grads_b_1d], 0)  # Concatenar gradientes de sesgos
        return loss_val.numpy(), grads_1d.numpy()
    
    def optimizer_callback(self, parameters):
        self.set_weights(parameters)
        self.iteration += 1  # Incrementar contador de iteraciones
        if self.iteration % 1 == 0:
            loss_value = self.loss(x0_tf, x1_tf, dt, IRK_weights, u0_tf)
            U1_pred = self.evaluate(x_star_tf)
            U1_pred = U1_pred.numpy()
            error = np.linalg.norm(U1_pred[:, -1] - Exact[idx_t1, :], 2) / np.linalg.norm(Exact[idx_t1, :], 2)
            print(f'Error relativo: {error}')
            tf.print(f'It: {self.iteration} Loss:{loss_value}')
            init_params = self.get_weights().numpy()
            np.save('init_params.npy', init_params)

    def second_derivative(self,func, x):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            y = func(x)
            dy_dx = tape.gradient(y, x)
        d2y_dx2 = tape.gradient(dy_dx, x)
        return d2y_dx2

    # def net_U0(self, x, dt, IRK_weights):
    #     with tf.GradientTape(persistent=True) as tape:
    #         tape.watch(x)
    #         U1 = self.evaluate(x)
    #         U = U1[:, :-1]
    #         U_x = tape.gradient(U, x)
    #     U_xx = tape.gradient(U_x, x)
    #     F = 5.0 * U - 5.0 * U ** 3 + 0.0001 * U_xx
    #     U0 = U1 - dt * tf.matmul(F, tf.transpose(IRK_weights))
    #     return U0
    def net_U0(self, x, dt, IRK_weights):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            U1 = self.evaluate(x)
            U = U1[:, :-1]
            U_x = tape.gradient(U, x)
        
        U_xx = self.second_derivative(self.evaluate, x)
        F = 5.0 * U - 5.0 * U ** 3 + 0.0001 * U_xx
        U0 = U1 - dt * tf.matmul(F, tf.transpose(IRK_weights))
        return U0


    def net_U1(self, x):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            U1 = self.evaluate(x)
        U1_x = tape.gradient(U1, x)
        return U1, U1_x                           
    
    def loss(self, x0_tf, x1_tf, dt, IRK_weights, u0_tf):
        self.U0_pred = self.net_U0(x0_tf, dt, IRK_weights)
        self.U1_pred, self.U1_x_pred = self.net_U1(x1_tf)
        loss = tf.reduce_sum(tf.square(u0_tf - self.U0_pred)) + \
               tf.reduce_sum(tf.square(self.U1_pred[0, :] - self.U1_pred[1, :])) + \
               tf.reduce_sum(tf.square(self.U1_x_pred[0, :] - self.U1_x_pred[1, :]))          
        return loss

# Configuración de parámetros
q = 100
layers = [1, 200, 200, 200, 200, q + 1]
lb = np.array([-1.0])
ub = np.array([1.0])
N = 200
noise_u0 = 0.0
idx_t0 = 20
idx_t1 = 180

# Carga de datos y preparación
tmp = np.float32(np.loadtxt('../../Utilities/IRK_weights/Butcher_IRK%d.txt' % q, ndmin=2))
IRK_weights = tf.constant(np.reshape(tmp[0:q ** 2 + q], (q + 1, q)), dtype=tf.float64)

data = scipy.io.loadmat('../Data/AC.mat')
t = data['tt'].flatten()[:, None]
x = data['x'].flatten()[:, None]
x_star = x
x1 = np.vstack((lb, ub))
Exact = np.real(data['uu']).T
dt = tf.constant(t[idx_t1] - t[idx_t0], dtype=tf.float64)
x1_tf = tf.Variable(x1, dtype=tf.float64)
x_star_tf = tf.Variable(x_star, dtype=tf.float64)
idx_x = np.random.choice(Exact.shape[1], N, replace=False)
x0 = x[idx_x, :]
u0 = Exact[idx_t0:idx_t0 + 1, idx_x].T
u0 = u0 + noise_u0 * np.std(u0) * np.random.randn(u0.shape[0], u0.shape[1])
x0_tf = tf.Variable(x0, dtype=tf.float64)
u0_tf = tf.Variable(u0, dtype=tf.float64)

# Inicialización del modelo
PINN = Sequentialmodel(layers)

# if os.path.isfile('init_params.npy'):
#     init_params = np.load('init_params.npy')
# else:
#     init_params = PINN.get_weights().numpy()

# # Entrenamiento con Adam
# def train_adam(model, x0_tf, x1_tf, dt, IRK_weights, u0_tf, learning_rate=1e-3, num_epochs=500):
#     optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
#     for epoch in range(num_epochs):
#         with tf.GradientTape() as tape:
#             loss_value = model.loss(x0_tf, x1_tf, dt, IRK_weights, u0_tf)
#         grads = tape.gradient(loss_value, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
#         if epoch % 100 == 0:
#             print(f"Adam Epoch {epoch}: Loss {loss_value.numpy()}")

# # Entrenamiento
# start_time_adam = time.time()
# train_adam(PINN, x0_tf, x1_tf, dt, IRK_weights, u0_tf, num_epochs=500)
# end_time_adam = time.time()
# adam_training_time = end_time_adam - start_time_adam
# print(f"Adam training time: {adam_training_time:.2f} seconds")

init_params = PINN.get_weights().numpy()

# Entrenamiento con L-BFGS
start_time_lbfgs = time.time()
results = scipy.optimize.minimize(fun=PINN.optimizerfunc, 
                                  x0=init_params, 
                                  args=(), 
                                  method='L-BFGS-B', 
                                  jac=True,
                                  callback=PINN.optimizer_callback, 
                                  options={'disp': None,
                                           'maxcor': 50, 
                                           'ftol': 1.0 * np.finfo(float).eps,
                                           'maxfun': 100_000, 
                                           'maxiter': 100_000,
                                           'maxls': 50})
end_time_lbfgs = time.time()
lbfgs_training_time = end_time_lbfgs - start_time_lbfgs
print(f"LBFGS training time: {lbfgs_training_time:.2f} seconds")

total_training_time = lbfgs_training_time#adam_training_time + lbfgs_training_time
print(f"Total training time: {total_training_time:.2f} seconds")

# Guardar los tiempos de entrenamiento en un archivo
with open('training_times.txt', 'w') as f:
    #f.write(f"Adam training time: {adam_training_time:.2f} seconds\n")
    f.write(f"LBFGS training time: {lbfgs_training_time:.2f} seconds\n")
    f.write(f"Total training time: {total_training_time:.2f} seconds\n")

# Evaluar el modelo
U1_pred = PINN.evaluate(x_star_tf).numpy()
error = np.linalg.norm(U1_pred[:, -1] - Exact[idx_t1, :], 2) / np.linalg.norm(Exact[idx_t1, :], 2)
print('Error: %e' % (error))
