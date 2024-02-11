""" Hebb-activation functions.

    :Version:
        2.0.0

    :Date:
        04.07.2018

    :Author:
        Jan Melchior

    :Contact:
        JanMelchior@gmx.de

    :License:

        Copyright (C) 2018 Jan Melchior

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
from pydeep.base.numpyextension import log_sum_exp

class HebbSigmoid(object):
    """ Hebb-Sigmoid function. Derivative is set to 1 so that the derivative of the Activation function is ignored
        during back prop. This leads to generalized Hebb learning/Oja's rule if Squared error objective is for aa single
        layer network.

        :Info: http://www.wolframalpha.com/input/?i=sigmoid

    """

    @classmethod
    def f(cls, x):
        """ Calculates the Sigmoid function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the Sigmoid function for x.
        :rtype: scalar or numpy array with the same shape as x.

        """
        return 0.5 + 0.5 * np.tanh(0.5 * x)

    @classmethod
    def g(cls, y):
        """ Calculates the inverse Sigmoid function value for a given input y.

        :param y: Input data.
        :type y: scalar or numpy array.

        :return: Value of the inverse Sigmoid function for y.
        :rtype: scalar or numpy array with the same shape as y.

        """
        return 2.0 * np.arctanh(2.0 * y - 1.0)

    @classmethod
    def df(cls, x):
        """ The derivative function is overwritten by 1.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Constant 1.0
        :rtype: scalar

        """
        return 1.0

    @classmethod
    def dg(cls, y):
        """ The second derivative function is overwritten by 1.

        :param y: Input data.
        :type y: scalar or numpy array

        :return: Constant 1.0
        :rtype: scalar

        """
        return 1.0

    @classmethod
    def ddf(cls, x):
        """ The second derivative function is overwritten by 1.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Constant 1.0
        :rtype: scalar

        """
        return 1.0


class HebbStep(object):
    """ Hebb-Sigmoid function. Derivative is set to 1 so that the derivative of the Activation function is ignored
        during back prop. This leads to generalized Hebb learning/Oja's rule if Squared error objective is for aa single
        layer network.

        :Info: http://www.wolframalpha.com/input/?i=sigmoid

    """

    @classmethod
    def f(cls, x):
        """ Calculates the Sigmoid function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the Sigmoid function for x.
        :rtype: scalar or numpy array with the same shape as x.

        """
        return np.float64(x > 0)

    @classmethod
    def df(cls, x):
        """ The derivative function is overwritten by 1.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Constant 1.0
        :rtype: scalar

        """
        return 1.0

    @classmethod
    def dg(cls, y):
        """ The second derivative function is overwritten by 1.

        :param y: Input data.
        :type y: scalar or numpy array

        :return: Constant 1.0
        :rtype: scalar

        """
        return 1.0

    @classmethod
    def ddf(cls, x):
        """ The second derivative function is overwritten by 1.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Constant 1.0
        :rtype: scalar

        """
        return 1.0


class HebbRectifier(object):
    """ Hebb-Rectifier function. Derivative is set to 1 so that the derivative of the Activation function is ignored
        during back prop. This leads to generalized Hebb learning/Oja's rule if Squared error objective is for aa single
        layer network.
    """

    @classmethod
    def f(cls, x):
        """ Calculates the Rectifier function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array.

        :return: Value of the Rectifier function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return np.maximum(0.0, x)


    @classmethod
    def df(cls, x):
        """ The derivative function is overwritten by 1.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Constant 1.0
        :rtype: scalar

        """
        return 1.0

    @classmethod
    def dg(cls, y):
        """ The second derivative function is overwritten by 1.

        :param y: Input data.
        :type y: scalar or numpy array

        :return: Constant 1.0
        :rtype: scalar

        """
        return 1.0

    @classmethod
    def ddf(cls, x):
        """ The second derivative function is overwritten by 1.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Constant 1.0
        :rtype: scalar

        """
        return 1.0


class HebbSoftMax(object):
    """ Soft Max function.

    """

    @classmethod
    def f(cls, x):
        """ Calculates the function value of the SoftMax function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Value of the SoftMax function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return np.exp(x - log_sum_exp(x, axis=1).reshape(x.shape[0], 1))

    @classmethod
    def df(cls, x):
        """ Calculates the derivative of the SoftMax function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Value of the derivative of the SoftMax function for x.
        :rtype: scalar or numpy array with the same shape as x.
        """
        return 1.0


class HebbKMax(object):
    """ Hebb-K-max function. Derivative is set to 1 so that the derivative of the Activation function is ignored
        during back prop. This leads to generalized Hebb learning/Oja's rule if Squared error objective is for aa single
        layer network.

    """

    def __init__(self, k, axis=1):
        """ Constructor.

        :param k: Number of active units.
        :type k: int

        :param axis: Axis to compute the maximum.
        :type axis: int

        """
        self.k = k
        self.axis = axis
        self._temp_a = None

    def f(self, x):
        """ Calculates the K-max function value for a given input x.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Value of the Kmax function for x.
        :rtype: scalar or numpy array with the same shape as x.

        """
        act = HebbSigmoid.f(np.atleast_2d(x))
        if self.axis == 0:
            self._temp_a = np.float64(act >= np.atleast_2d(np.sort(act, axis=self.axis)[-self.k, :]))
        else:
            self._temp_a = np.float64(act.T >= np.atleast_2d(np.sort(act, axis=self.axis)[:, -self.k])).T
        return act*self._temp_a

    def df(self, x):
        """ The derivative function is overwritten by 1.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Constant 1.0
        :rtype: scalar

        """
        return 1.0

    def ddf(self, x):
        """ The second derivative function is overwritten by 1.

        :param x: Input data.
        :type x: scalar or numpy array

        :return: Constant 1.0
        :rtype: scalar

        """
        return 1.0
