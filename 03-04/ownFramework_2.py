
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.cm as cm
import pprint
import matplotlib.pyplot as plt
plt.ion()  # Habilita el modo interactivo
n = 100
X, Y = make_classification(n_samples = n, n_features=2,
                           n_redundant=0, n_informative=2, flip_y=0.2)
X = X.astype(np.float32)
Y = Y.astype(np.int32)

# Split into train and test dataset
train_x, test_x = np.split(X, [n*8//10])
train_labels, test_labels = np.split(Y, [n*8//10])

class CrossEntropyLoss:
    def forward(self,p,y):
        self.p = p
        self.y = y
        p_of_y = p[np.arange(len(y)), y]
        log_prob = np.log(p_of_y)
        return -log_prob.mean() # average over all input samples

    def backward(self, loss):
        dlog_softmax = np.zeros_like(self.p)
        dlog_softmax[np.arange(len(self.y)), self.y] -= 1.0 / len(self.y)
        return dlog_softmax / self.p

def train_epoch(net, train_x, train_labels, loss=CrossEntropyLoss(), batch_size=4, lr=0.1):
    for i in range(0, len(train_x), batch_size):
        xb = train_x[i:i + batch_size]
        yb = train_labels[i:i + batch_size]

        p = net.forward(xb)
        l = loss.forward(p, yb)
        dp = loss.backward(l)
        dx = net.backward(dp)
        net.update(lr)


class Net:
    def __init__(self):
        self.layers = []

    def add(self, l):
        self.layers.append(l)

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, z):
        for l in self.layers[::-1]:
            z = l.backward(z)
        return z

    def update(self, lr):
        for l in self.layers:
            if 'update' in l.__dir__():
                l.update(lr)

def get_loss_acc(x,y,loss=CrossEntropyLoss()):
    p = net.forward(x)
    l = loss.forward(p,y)
    pred = np.argmax(p,axis=1)
    acc = (pred==y).mean()
    return l,acc

def plot_training_progress(x, y_data, fig, ax):
    styles = ['k--', 'g-']
    # remove previous plot
    while ax.lines:
        if ax.lines:
            ax.lines.pop()
    # draw updated lines
    for i in range(len(y_data)):
        ax.plot(x, y_data[i], styles[i])
    ax.legend(ax.lines, ['training accuracy', 'validation accuracy'],
              loc='upper center', ncol = 2)

def train_and_plot(n_epoch, net, loss=CrossEntropyLoss(), batch_size=4, lr=0.1):
    fig, ax = plt.subplots(2, 1)
    ax[0].set_xlim(0, n_epoch + 1)
    ax[0].set_ylim(0, 1)

    train_acc = np.empty((n_epoch, 3))
    train_acc[:] = np.nan
    valid_acc = np.empty((n_epoch, 3))
    valid_acc[:] = np.nan

    for epoch in range(1, n_epoch + 1):
        train_epoch(net, train_x, train_labels, loss, batch_size, lr)
        tloss, taccuracy = get_loss_acc(train_x, train_labels, loss)
        train_acc[epoch - 1, :] = [epoch, tloss, taccuracy]
        vloss, vaccuracy = get_loss_acc(test_x, test_labels, loss)
        valid_acc[epoch - 1, :] = [epoch, vloss, vaccuracy]

        ax[0].set_ylim(0, max(max(train_acc[:, 2]), max(valid_acc[:, 2])) * 1.1)

        plot_training_progress(train_acc[:, 0], (train_acc[:, 2], valid_acc[:, 2]), fig, ax[0])
        plot_decision_boundary(net, fig, ax[1])
        fig.canvas.draw()
        fig.canvas.flush_events()

    return train_acc, valid_acc


def plot_decision_boundary(net, fig, ax):
    draw_colorbar = True
    # remove previous plot
    while ax.collections:
        #ax.collections.pop()
        line = ax.collections[0]
        line.remove()
        draw_colorbar = False

    # generate countour grid
    x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 1
    y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid_points = np.c_[xx.ravel().astype('float32'), yy.ravel().astype('float32')]
    n_classes = max(train_labels) + 1
    while train_x.shape[1] > grid_points.shape[1]:
        # pad dimensions (plot only the first two)
        grid_points = np.c_[grid_points,
        np.empty(len(xx.ravel())).astype('float32')]
        grid_points[:, -1].fill(train_x[:, grid_points.shape[1] - 1].mean())

    # evaluate predictions
    prediction = np.array(net.forward(grid_points))
    # for two classes: prediction difference
    if (n_classes == 2):
        Z = np.array([0.5 + (p[0] - p[1]) / 2.0 for p in prediction]).reshape(xx.shape)
    else:
        Z = np.array([p.argsort()[-1] / float(n_classes - 1) for p in prediction]).reshape(xx.shape)

    # draw contour
    levels = np.linspace(0, 1, 40)
    cs = ax.contourf(xx, yy, Z, alpha=0.4, levels=levels)
    if draw_colorbar:
        fig.colorbar(cs, ax=ax, ticks=[0, 0.5, 1])
    c_map = [cm.jet(x) for x in np.linspace(0.0, 1.0, n_classes)]
    colors = [c_map[l] for l in train_labels]
    ax.scatter(train_x[:, 0], train_x[:, 1], marker='o', c=colors, s=60, alpha=0.5)

def plot_training_progress(x, y_data, fig, ax):
    styles = ['k--', 'g-']
    # remove previous plot
    while ax.lines:
#        pprint.pprint(ax.lines)
#        exit()
        #ax.lines.pop()
        line = ax.lines[0]
        line.remove()
    # draw updated lines
    for i in range(len(y_data)):
        ax.plot(x, y_data[i], styles[i])
    ax.legend(ax.lines, ['training accuracy', 'validation accuracy'],
              loc='upper center', ncol = 2)


class Linear:
    def __init__(self, nin, nout):
        self.W = np.random.normal(0, 1.0 / np.sqrt(nin), (nout, nin))
        self.b = np.zeros((1, nout))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W.T) + self.b

    def backward(self, dz):
        dx = np.dot(dz, self.W)
        dW = np.dot(dz.T, self.x)
        db = dz.sum(axis=0)
        self.dW = dW
        self.db = db
        return dx

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class Softmax:
    def forward(self, z):
        self.z = z
        zmax = z.max(axis=1, keepdims=True)
        expz = np.exp(z - zmax)
        Z = expz.sum(axis=1, keepdims=True)
        return expz / Z

    def backward(self, dp):
        p = self.forward(self.z)
        pdp = p * dp
        return pdp - p * pdp.sum(axis=1, keepdims=True)


class CrossEntropyLoss:
    def forward(self, p, y):
        self.p = p
        self.y = y
        p_of_y = p[np.arange(len(y)), y]
        log_prob = np.log(p_of_y)
        return -log_prob.mean()

    def backward(self, loss):
        dlog_softmax = np.zeros_like(self.p)
        dlog_softmax[np.arange(len(self.y)), self.y] -= 1.0 / len(self.y)
        return dlog_softmax / self.p

net = Net()
net.add(Linear(2,2))
net.add(Softmax())

res = train_and_plot(30,net,lr=0.005)

lin = Linear(2, 2)
softmax = Softmax()
cross_ent_loss = CrossEntropyLoss()

learning_rate = 0.1

pred = np.argmax(lin.forward(train_x), axis=1)
acc = (pred == train_labels).mean()
print("Initial accuracy: ", acc)

batch_size = 4
for i in range(0, len(train_x), batch_size):
    xb = train_x[i:i + batch_size]
    yb = train_labels[i:i + batch_size]

    # forward pass
    z = lin.forward(xb)
    p = softmax.forward(z)
    loss = cross_ent_loss.forward(p, yb)

    # backward pass
    dp = cross_ent_loss.backward(loss)
    dz = softmax.backward(dp)
    dx = lin.backward(dz)
    lin.update(learning_rate)

pred = np.argmax(lin.forward(train_x), axis=1)
acc = (pred == train_labels).mean()
print("Final accuracy: ", acc)


class Net:
    def __init__(self):
        self.layers = []

    def add(self, l):
        self.layers.append(l)

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, z):
        for l in self.layers[::-1]:
            z = l.backward(z)
        return z

    def update(self, lr):
        for l in self.layers:
            if 'update' in l.__dir__():
                l.update(lr)


net = Net()
net.add(Linear(2, 2))
net.add(Softmax())
loss = CrossEntropyLoss()


def get_loss_acc(x, y, loss=CrossEntropyLoss()):
    p = net.forward(x)
    l = loss.forward(p, y)
    pred = np.argmax(p, axis=1)
    acc = (pred == y).mean()
    return l, acc


print("Initial loss={}, accuracy={}: ".format(*get_loss_acc(train_x, train_labels)))


def train_epoch(net, train_x, train_labels, loss=CrossEntropyLoss(), batch_size=4, lr=0.1):
    for i in range(0, len(train_x), batch_size):
        xb = train_x[i:i + batch_size]
        yb = train_labels[i:i + batch_size]

        p = net.forward(xb)
        l = loss.forward(p, yb)
        dp = loss.backward(l)
        dx = net.backward(dp)
        net.update(lr)


train_epoch(net, train_x, train_labels)

print("Final loss={}, accuracy={}: ".format(*get_loss_acc(train_x, train_labels)))
print("Test loss={}, accuracy={}: ".format(*get_loss_acc(test_x, test_labels)))


def train_and_plot(n_epoch, net, loss=CrossEntropyLoss(), batch_size=4, lr=0.1):
    fig, ax = plt.subplots(2, 1)
    ax[0].set_xlim(0, n_epoch + 1)
    ax[0].set_ylim(0, 1)

    train_acc = np.empty((n_epoch, 3))
    train_acc[:] = np.nan
    valid_acc = np.empty((n_epoch, 3))
    valid_acc[:] = np.nan

    for epoch in range(1, n_epoch + 1):
        train_epoch(net, train_x, train_labels, loss, batch_size, lr)
        tloss, taccuracy = get_loss_acc(train_x, train_labels, loss)
        train_acc[epoch - 1, :] = [epoch, tloss, taccuracy]
        vloss, vaccuracy = get_loss_acc(test_x, test_labels, loss)
        valid_acc[epoch - 1, :] = [epoch, vloss, vaccuracy]

        ax[0].set_ylim(0, max(max(train_acc[:, 2]), max(valid_acc[:, 2])) * 1.1)

        plot_training_progress(train_acc[:, 0], (train_acc[:, 2],
                                                 valid_acc[:, 2]), fig, ax[0])
        plot_decision_boundary(net, fig, ax[1])
        fig.canvas.draw()
        fig.canvas.flush_events()

    return train_acc, valid_acc


import matplotlib.cm as cm


def plot_decision_boundary(net, fig, ax):
    draw_colorbar = True
    # remove previous plot
    while ax.collections:
        #ax.collections.pop()
        line = ax.collections[0]
        line.remove()
        draw_colorbar = False

    # generate countour grid
    x_min, x_max = train_x[:, 0].min() - 1, train_x[:, 0].max() + 1
    y_min, y_max = train_x[:, 1].min() - 1, train_x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid_points = np.c_[xx.ravel().astype('float32'), yy.ravel().astype('float32')]
    n_classes = max(train_labels) + 1
    while train_x.shape[1] > grid_points.shape[1]:
        # pad dimensions (plot only the first two)
        grid_points = np.c_[grid_points,
        np.empty(len(xx.ravel())).astype('float32')]
        grid_points[:, -1].fill(train_x[:, grid_points.shape[1] - 1].mean())

    # evaluate predictions
    prediction = np.array(net.forward(grid_points))
    # for two classes: prediction difference
    if (n_classes == 2):
        Z = np.array([0.5 + (p[0] - p[1]) / 2.0 for p in prediction]).reshape(xx.shape)
    else:
        Z = np.array([p.argsort()[-1] / float(n_classes - 1) for p in prediction]).reshape(xx.shape)

    # draw contour
    levels = np.linspace(0, 1, 40)
    cs = ax.contourf(xx, yy, Z, alpha=0.4, levels=levels)
    if draw_colorbar:
        fig.colorbar(cs, ax=ax, ticks=[0, 0.5, 1])
    c_map = [cm.jet(x) for x in np.linspace(0.0, 1.0, n_classes)]
    colors = [c_map[l] for l in train_labels]
    ax.scatter(train_x[:, 0], train_x[:, 1], marker='o', c=colors, s=60, alpha=0.5)

def plot_training_progress(x, y_data, fig, ax):
    styles = ['k--', 'g-']
    # remove previous plot
    while ax.lines:
        #ax.lines.pop()
        line = ax.lines[0]
        line.remove()
    # draw updated lines
    for i in range(len(y_data)):
        ax.plot(x, y_data[i], styles[i])
    ax.legend(ax.lines, ['training accuracy', 'validation accuracy'],
              loc='upper center', ncol = 2)

#%matplotlib nbagg
net = Net()
net.add(Linear(2,2))
net.add(Softmax())

res = train_and_plot(30,net,lr=0.005)
plt.show(block=True)