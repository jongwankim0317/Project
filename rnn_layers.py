"""
* Copyright 2020 Jongwankim
* Released under the MIT license.
* http://github.com/jongwankim0317
"""

import numpy as np

def rnn_step_forward(x, prev_h, Wx, Wh, b):

	tmp_matmul = np.matmul(x, Wx) + np.matmul(prev_h, Wh) + b
	next_h = np.tanh(tmp_matmul)
	cache = (x, prev_h, Wx, Wh, b, next_h)

	return next_h, cache


def rnn_step_backward(dnext_h, cache):
	dx, dprev_h, dWx, dWh, db = None, None, None, None, None
	(x, prev_h, Wx, Wh, b, next_h) = cache

	tmp_matmul = np.matmul(x, Wx) + np.matmul(prev_h, Wh) + b
	next_h = np.tanh(tmp_matmul)

	next_h_grad = np.ones(shape=next_h.shape) - next_h * next_h

	dout = next_h_grad * dnext_h

	dx = np.matmul(dout, Wx.T)
	dprev_h = np.matmul(dout, Wh.T)
	dWx = np.matmul(x.T, dout)
	dWh = np.matmul(prev_h.T, dout)

	b_grad = np.ones(shape=(dout.shape[0]))
	db = np.matmul(b_grad, dout)

	return dx, dprev_h, dWx, dWh, db


#------------------------------------------------------
# vanilla rnn forward
#------------------------------------------------------

def rnn_forward(x, h0, Wx, Wh, b):

	h, cache = None, None
	N, T, D = x.shape
	H = Wh.shape[0]

	h_stack = np.zeros(shape=(N, T, H))

	cache = []
	th = h0
	for t in range(T):
		tx = x[:, t, :]  # tx¸¦ (N,H) shapeÀÇ T°³ÀÇ º¤ÅÍÁß ÇÑ°³¾¿ ¿ø¼Ò¸¦ »ÌÀ½
		th_, cache_ = rnn_step_forward(tx, th, Wx, Wh, b)
		th = th_  # hidden state¸¦ update

		h_stack[:, t, :] = th  # updateµÈ hidden state¸¦ T°³ÀÇ ±æÀÌ¸¦ °¡Áø º¤ÅÍ·Î ¸¸µë
		# th.shape = (N,H) -> h_stack.shape = (N,T,H)
		cache.append(cache_)

	h = h_stack
	return h, cache


#------------------------------------------------------
# vanilla rnn backward 
#------------------------------------------------------

def rnn_backward(dh, cache):

	dx, dh0, dWx, dWh, db = None, None, None, None, None

	N, T, H = dh.shape
	D = cache[0][0].shape[1]

	dx = np.zeros((N, T, D))
	dh0 = np.zeros((N, H))
	dWx = np.zeros((D, H))
	dWh = np.zeros((H, H))
	db = np.zeros((H))
	dprev_h = np.zeros((N, H))

	cache_ = cache

	# print(len(cache))
	# print(cache[0])
	# print(cache[0][0])

	for t in range(T - 1, -1, -1):
		current_dh = dh[:, t, :] + dprev_h
		cache_ = cache[t]
		dx_, dprev_h, dWx_, dWh_, db_ = rnn_step_backward(current_dh, cache_)

		dx[:, t, :] = dx_
		dWx += dWx_
		dWh += dWh_
		db += db_

	dh0 = dprev_h

	return dx, dh0, dWx, dWh, db



def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def getdata_rnn_step_forward():

	np.random.seed(2177)

	N, D, H = 3, 5, 4
	x = np.random.randn(N, D)
	prev_h = np.random.randn(N, H)
	Wx = np.random.randn(D, H)
	Wh = np.random.randn(H, H)
	b = np.random.randn(H)

	expt_next_h = np.asarray([
		[-0.99921173, -0.99967951,  0.39127099, -0.93436299],
	 	[ 0.84348286,  0.99996526, -0.9978802,   0.99996645],
		[-0.94481752, -0.71940178,  0.99994009, -0.64806562]])

	return x, prev_h, Wx, Wh, b, expt_next_h


def getdata_rnn_step_backward(x,h,Wx,Wh,b,dnext_h):

	fx  = lambda x: rnn_step_forward(x, h, Wx, Wh, b)[0]
	fh  = lambda prev_h: rnn_step_forward(x, h, Wx, Wh, b)[0]
	fWx = lambda Wx: rnn_step_forward(x, h, Wx, Wh, b)[0]
	fWh = lambda Wh: rnn_step_forward(x, h, Wx, Wh, b)[0]
	fb  = lambda b: rnn_step_forward(x, h, Wx, Wh, b)[0]

	dx_num = eval_numerical_gradient_array(fx, x, dnext_h)
	dprev_h_num = eval_numerical_gradient_array(fh, h, dnext_h)
	dWx_num = eval_numerical_gradient_array(fWx, Wx, dnext_h)
	dWh_num = eval_numerical_gradient_array(fWh, Wh, dnext_h)
	db_num = eval_numerical_gradient_array(fb, b, dnext_h)

	return dx_num, dprev_h_num, dWx_num, dWh_num, db_num


def getdata_rnn_forward():
	np.random.seed(2177)

	N, D, T, H = 2,3,4,5
	x = np.random.randn(N, T, D)
	h0 = np.random.randn(N, H)
	Wx = np.random.randn(D, H)
	Wh = np.random.randn(H, H)
	b = np.random.randn(H)

	expt_next_h = np.asarray([
	[[ 0.79899136, -0.90076473, -0.69325878, -0.99991011,  0.92991908],
	 [-0.04474799, -0.99999994, -0.72167573, -0.99942462, -0.98397185],
	 [ 0.98674954, -0.74668554, -0.30836793, -0.87580427, -0.25076433],
	 [ 0.99999994,  0.46495278, -0.6291276 ,  0.44811995, -0.91013617]],

 	[[-0.57789921, -0.10875688, -0.99049558, -0.58448393,  0.76942269],
	 [-0.05646372, -0.99855467, -0.827688  , -0.65262183, -0.98211725],
	 [ 0.89687939,  0.99998112, -0.99999517,  0.66932722,  0.99952606],
	 [-0.97608409, -0.64972242, -0.99987169, -0.99747724,  0.99962792]]])

	return x, h0, Wx, Wh, b, expt_next_h

def getdata_rnn_backward(x,h0,Wx,Wh,b,dout):
	fx = lambda x: rnn_forward(x, h0, Wx, Wh, b)[0]
	fh0 = lambda h0: rnn_forward(x, h0, Wx, Wh, b)[0]
	fWx = lambda Wx: rnn_forward(x, h0, Wx, Wh, b)[0]
	fWh = lambda Wh: rnn_forward(x, h0, Wx, Wh, b)[0]
	fb = lambda b: rnn_forward(x, h0, Wx, Wh, b)[0]

	dx_num = eval_numerical_gradient_array(fx, x, dout)
	dh0_num = eval_numerical_gradient_array(fh0, h0, dout)
	dWx_num = eval_numerical_gradient_array(fWx, Wx, dout)
	dWh_num = eval_numerical_gradient_array(fWh, Wh, dout)
	db_num = eval_numerical_gradient_array(fb, b, dout)
	
	return dx_num, dh0_num, dWx_num, dWh_num, db_num


