from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        pre = input_dim
        for i in range(len(hidden_dims)):
            self.params["W"+str(i+1)] = weight_scale * \
                np.random.randn(pre, hidden_dims[i])
            self.params["b"+str(i+1)] = np.zeros(hidden_dims[i])
            pre = hidden_dims[i]
            if self.normalization is not None:
                self.params["gamma"+str(i+1)] = np.ones(pre)
                self.params["beta"+str(i+1)] = np.zeros(pre)
                # L-1层后最后输出层
        self.params["W"+str(1+len(hidden_dims))] = weight_scale * \
            np.random.randn(pre, num_classes)
        self.params["b"+str(1+len(hidden_dims))] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"}
                              for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        x_pre = X  # 记录上一结点的结果
        cache = {}  # 记录每个结点的x,w,b,gamma,beta
        cache_dropout = {}
        for i in range(self.num_layers-1):
            W, b = self.params["W"+str(i+1)], self.params["b"+str(i+1)]
            if self.normalization == "batchnorm":
                gamma = self.params["gamma" + str(i+1)]
                beta = self.params["beta"+str(i+1)]
                bn_param = self.bn_params[i]
                x_pre, cache[i] = affine_bn_relu_forward(
                    x_pre, W, b, gamma, beta, bn_param)
            elif self.normalization == "layernorm":
                gamma = self.params["gamma" + str(i+1)]
                beta = self.params["beta"+str(i+1)]
                bn_param = self.bn_params[i]
                x_pre, cache[i] = affine_ln_relu_forward(
                    x_pre, W, b, gamma, beta, bn_param)
            else:
                x_pre, cache[i] = affine_relu_forward(x_pre, W, b)
            if self.use_dropout:
                x_pre, cache_dropout[i] = dropout_forward(
                    x_pre, self.dropout_param)

        # 最后一个输出层
        i = self.num_layers-1
        W, b = self.params["W"+str(i+1)], self.params["b"+str(i+1)]
        scores, cache[i] = affine_forward(x_pre, W, b)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dout = softmax_loss(scores, y)
        for i in range(self.num_layers):
            loss += 0.5*self.reg * \
                np.sum(self.params["W"+str(i+1)]*self.params["W"+str(i+1)])
        # 处理最后一层
        i = self.num_layers-1
        loss += 0.5*self.reg * \
            np.sum(self.params["W"+str(i+1)]*self.params["W"+str(i+1)])
        dx, dw, db = affine_backward(dout, cache[i])
        # 这里疑似有错误，在前面reg*2后notebook的测试中W3就是对的，但事实上如果L2乘了0.5这里就没有2（我估计是他的验证代码的问题）
        grads["W"+str(i+1)] = dw+self.reg*self.params["W"+str(i+1)]
        grads["b"+str(i+1)] = db
        for i in range(self.num_layers-1, 0, -1):
            if self.use_dropout:
                dx = dropout_backward(dx, cache_dropout[i-1])
            if self.normalization == "batchnorm":
                dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(
                    dx, cache[i-1])
                grads["gamma"+str(i)] = dgamma
                grads["beta"+str(i)] = dbeta
            elif self.normalization == "layernorm":
                dx, dw, db, dgamma, dbeta = affine_ln_relu_backward(
                    dx, cache[i-1])
                grads["gamma"+str(i)] = dgamma
                grads["beta"+str(i)] = dbeta
            else:
                dx, dw, db = affine_relu_backward(dx, cache[i-1])
            grads["W"+str(i)] = dw+self.reg*self.params["W"+str(i)]
            grads["b"+str(i)] = db

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads