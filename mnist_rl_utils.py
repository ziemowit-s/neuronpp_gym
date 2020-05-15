import tensorflow as tf
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import cv2


class AgentParam():
    def __init__(self, name, stepsize, label_num, skip, input_num, dt, rest_potential, out_labels):
        self.name = name
        self.stepsize = stepsize
        self.label_num = label_num
        self.skip = skip
        self.input_num = input_num
        self.dt = dt
        self.rest_potential = rest_potential
        self.out_labels = out_labels


def get_cell_weights(cells: list, weight_name: str = "w"):
    weights = []
    for cell in cells:
        for point_process in cell.pps:
            try:
                weight = getattr(point_process.hoc, weight_name)
                weights.append(weight)
            except AttributeError:
                pass
    return np.array(weights)


class LinApprox():
    def __init__(self, state_len, act_no):
        self.state_len = state_len
        self.act_no = act_no
        self.poly = PolynomialFeatures(2)
        self.method = 'stack'
        self.processed_len = self.poly.fit(np.ones(self.state_len).reshape(1, -1)).n_output_features_
        # todo initialization of theta? see coursera: optimistic? pessimistic?
        self.total_input_length = self.processed_len * self.act_no + 1
        self.theta = np.zeros(self.total_input_length).reshape(-1, 1)
        self.theta[-1] = 1.0  # bias weight
        # eligibility trace
        self.z = np.ones(self.total_input_length)

    def q_val(self, state, action):
        val = self.phi_state_action(state=state, action=action).dot(self.theta)
        return val

    def phi_state(self, state):
        phi = self.poly.fit_transform(state.reshape(1, -1))
        return phi

    def phi_state_action(self, state, action):
        tmp = self.phi_state(state)
        phi_state_action = np.zeros(self.total_input_length)
        phi_state_action[action * self.processed_len:(action + 1) * self.processed_len] = tmp
        phi_state_action[-1] = 1.0
        return phi_state_action

    def update_eligibility(self, state, action, gamma, lambda_par):
        self.z = self.phi_state_action(state=state, action=action) + gamma * lambda_par * self.z

    def update_theta(self, alpha, delta):
        self.theta += alpha * delta * self.z


def get_predicted_reward(output, true):
    predicted = -1
    reward = -1
    # todo if len() == 0, then probably there is no need to check the value?
    if len(output) == 1 and output[0].value > -1:
        predicted = output[0].index
        if predicted == true:
            reward = +1
    return predicted, reward


def mnist_prepare(num=10):
    # todo change to numpy procedure
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    index_list = []
    for i, labels in enumerate(y_train):
        if labels in list(np.arange(num)):
            index_list.append(i)

    x_train, y_train = x_train[index_list], y_train[index_list]
    return x_train, y_train


def get_observation(X, idx):
    return cv2.normalize(cv2.resize(X[idx], (14, 14)), None, 0, 1.0, cv2.NORM_MINMAX)


# todo rewrite to numpy.metrics
class QueueNP:
    def __init__(self, class_no: int, length: int = 250, alpha: float = 0.9, gamma: float = 0.99):
        self._queue = np.zeros(shape=length, dtype=int)
        self._length = length
        self._pushed = 0
        self._qval = 0
        self._alpha = alpha
        self._gamma = gamma
        self._class_no = class_no
        self._tp = 0
        self._fp = 1
        self._tn = 2
        self._fn = 3
        self._tfpn = np.zeros((class_no, 4))
        self._mcc = np.zeros(class_no)
        self._f1 = np.zeros(class_no)

    def push(self, r: float, true: int, pred: int, qval: float):
        self._queue[:-1] = self._queue[1:]
        self._queue[-1] = r
        self._pushed = np.min((self._pushed + 1, self._length))
        # todo perhaps modify to compute the qval inside the queue (?)
        self._qval = qval
        for k in range(self._class_no):
            if k == true:
                if pred == k:
                    self._tfpn[k][self._tp] += 1
                else:
                    self._tfpn[k][self._fn] += 1
            else:
                if pred == k:
                    self._tfpn[k][self._fp] += 1
                else:
                    self._tfpn[k][self._tn] += 1
        for k in range(self._class_no):
            try:
                self._f1[k] = 2 * self._tfpn[k][self._tp] / (
                        2 * self._tfpn[k][self._tp] + self._tfpn[k][self._fp] + self._tfpn[k][self._fn])
            except RuntimeWarning:
                self._f1[k] = 0
            # todo add exception for zero-value denominator
            try:
                self._mcc[k] = (self._tfpn[k][self._tp] * self._tfpn[k][self._tn] -
                                self._tfpn[k][self._fp] * self._tfpn[k][self._fn]) \
                               / np.sqrt((self._tfpn[k][self._tp] + self._tfpn[k][self._fp]) *
                                         (self._tfpn[k][self._tp] + self._tfpn[k][self._fn]) *
                                         (self._tfpn[k][self._tn] + self._tfpn[k][self._fp]) *
                                         (self._tfpn[k][self._tn] + self._tfpn[k][self._fn]))
            except RuntimeWarning:
                self._mcc[k] = 0

    def sum(self):
        return np.sum(self._queue[-self.pushed:])

    def ratio(self):
        if self.pushed == 0:
            return 0.0
        else:
            return self.sum() / self.pushed

    @property
    def length(self):
        return self._length

    @property
    def pushed(self):
        return self._pushed

    @property
    def queue(self):
        return self._queue

    @property
    def qval(self):
        return self._qval

    @property
    def f1(self):
        return self._f1

    @property
    def mcc(self):
        return self._mcc
