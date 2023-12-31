from typing import Optional, Tuple, Union
import torch as t
from torch import Tensor
from jaxtyping import Float
import einops

def conv1d_minimal_simple(x: Float[Tensor, "w"], weights: Float[Tensor, "kw"]) -> Float[Tensor, "ow"]:
    w = x.shape[0]
    kw = weights.shape[0]
    ow = w - kw + 1
    s_w = x.stride(0)

    new_shape = (ow, kw)
    new_strides = (s_w, s_w)

    xs = x.as_strided(new_shape, new_strides)
    return einops.einsum(xs, weights, "ow kw, kw -> ow" )


# h = 10
# kernel_size = 3
# x = t.randn((h,))
# weights = t.randn((kernel_size,))
# my_output = conv1d_minimal_simple(x, weights)


# b = 6
# h = 15
# ci = 3
# co = 4
# kernel_size = 3
# x = t.randn((b, ci, h))
# weights = t.randn((co, ci, kernel_size))


def conv1d_minimal(x: Float[Tensor, "b ic w"], weights: Float[Tensor, "oc ic kw"]) -> Float[Tensor, "b oc ow"]:
    b, ic, w = x.shape
    oc, ic, kw = weights.shape
    ow = w - kw + 1
    s_b, s_ic, s_w = x.stride()
    new_shape = (b, ic, ow, kw)
    new_stride = (s_b, s_ic, s_w, s_w)
    x_strided = x.as_strided(size=new_shape, stride=new_stride)

    return einops.einsum(x_strided, weights, "b ic ow kw, oc ic kw -> b oc ow")

# my_output = conv1d_minimal(x, weights)


def conv2d_minimal(x: Float[Tensor, "b ic h w"], weights: Float[Tensor, "oc ic kh kw"]) -> Float[Tensor, "b oc oh ow"]:
    b, ic, h, w = x.shape
    oc, ic, kh, kw = weights.shape
    oh = h - kh + 1
    ow = w - kw + 1
    s_b, s_ic, s_h, s_w = x.stride()
    new_shape = (b, ic, oh, ow, kh, kw)
    new_stride = (s_b, s_ic, s_h, s_w, s_h, s_w)
    x_strided = x.as_strided(size=new_shape, stride=new_stride)

    return einops.einsum(x_strided, weights, "b ic oh ow kh kw, oc ic kh kw -> b oc oh ow")

# b = 6
# h = 48
# w = 64
# ci = 3
# co = 1
# kernel_size = 3, 3
# x = t.randn((b, ci, h, w), dtype=t.float64)
# weights = t.randn((co, ci, *kernel_size), dtype=t.float64)
# my_output = conv2d_minimal(x, weights)
# torch_output = t.conv2d(x, weights)
# t.testing.assert_close(my_output, torch_output)

def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    '''Pad a 1D tensor.'''
    b, ic, w = x.shape
    out = x.new_full((b, ic, w + left + right), pad_value)
    out[..., left:left+w] = x
    return out

# x = t.arange(4).float().view((1, 1, 4))
# actual = pad1d(x, 1, 3, -2.0)

def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
	B, C, H, W = x.shape
	output = x.new_full(size=(B, C, top + H + bottom, left + W + right), fill_value=pad_value)
	output[..., top : top + H, left : left + W] = x
	return output


def conv1d(
	x: Float[Tensor, "b ic w"],
	weights: Float[Tensor, "oc ic kw"],
	stride: int = 1,
	padding: int = 0
) -> Float[Tensor, "b oc ow"]:
    x_padded = pad1d(x, padding, padding, 0.0)

    b, ic, w = x_padded.shape
    oc, ic2, kw = weights.shape
    assert ic == ic2, "input channels must match"
    ow = 1 + (w - kw) // stride

    s_b, s_ic, s_w = x_padded.stride()
    new_shape = (b, ic, ow, kw)
    new_stride = (s_b, s_ic, stride * s_w, s_w)
    x_strided = x_padded.as_strided(size=new_shape, stride=new_stride)

    return einops.einsum(x_strided, weights, "b ic ow kw, oc ic kw -> b oc ow")

# b = 6
# h = 48
# w = 64
# ci = 3
# co = 1
# stride = 2
# padding = 1
# kernel_size = 3
# x = t.randn((b, ci, h))
# weights = t.randn((co, ci, kernel_size))
# my_output = conv1d(x, weights, stride=stride, padding=padding)
# torch_output = t.conv1d(x, weights, stride=stride, padding=padding)
# t.testing.assert_close(my_output, torch_output, atol=1e-4, rtol=1e-4)

IntOrPair = Union[int, Tuple[int, int]]
Pair = Tuple[int, int]
def force_pair(v: IntOrPair) -> Pair:
	'''Convert v to a pair of int, if it isn't already.'''
	if isinstance(v, tuple):
		if len(v) != 2:
			raise ValueError(v)
		return (int(v[0]), int(v[1]))
	elif isinstance(v, int):
		return (v, v)
	raise ValueError(v)


def conv2d(
	x: Float[Tensor, "b ic h w"],
	weights: Float[Tensor, "oc ic kh kw"],
	stride: IntOrPair = 1,
	padding: IntOrPair = 0
) -> Float[Tensor, "b oc oh ow"]:
    stride_h, stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)
    x_padded = pad2d(x, padding_w, padding_w, padding_h, padding_h, 0.0)

    b, ic, h, w = x_padded.shape
    oc, ic2, kh, kw = weights.shape
    assert ic == ic2, "input channels must match"
    oh = 1 + (h - kh) // stride_h
    ow = 1 + (w - kw) // stride_w

    s_b, s_ic, s_h, s_w = x_padded.stride()
    new_shape = (b, ic, oh, ow, kh, kw)
    new_stride = (s_b, s_ic, stride_h * s_h, stride_w * s_w, s_h, s_w)
    x_strided = x_padded.as_strided(size=new_shape, stride=new_stride)

    return einops.einsum(x_strided, weights, "b ic oh ow kh kw, oc ic kh kw -> b oc oh ow")

# x = t.randn((b, ci, h, w), dtype=t.float64)
# kernel_size = 3, 3
# weights = t.randn((co, ci, *kernel_size), dtype=t.float64)

# my_output = conv2d(x, weights, stride=stride, padding=padding)
# torch_output = t.conv2d(x, weights, stride=stride, padding=padding)
# t.testing.assert_close(my_output, torch_output)

def maxpool2d(
	x: Float[Tensor, "b ic h w"],
	kernel_size: IntOrPair,
	stride: Optional[IntOrPair] = None,
	padding: IntOrPair = 0
) -> Float[Tensor, "b ic oh ow"]:

    if stride is None:
        stride = kernel_size
    stride_h, stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)
    kh, kw = force_pair(kernel_size)
    x_padded = pad2d(x, left=padding_w, right=padding_w, top=padding_h, bottom=padding_h, pad_value=-t.inf)

    b, ic, h, w = x_padded.shape
    oh = 1 + (h - kh) // stride_h
    ow = 1 + (w - kw) // stride_w

    s_b, s_c, s_h, s_w = x_padded.stride()
    new_shape = (b, ic, oh, ow, kh, kw)
    new_stride = (s_b, s_c, s_h * stride_h, s_w * stride_w, s_h, s_w)

    x_strided = x_padded.as_strided(size=new_shape, stride=new_stride)

#    return t.amax(x_strided, dim=(-1, -2))
    return einops.reduce(x_strided, "b c oh ow kh kw -> b c oh ow", 'max')

# b = 6
# h = 48
# w = 64
# ci = 3
# co = 1
# stride = 2
# padding = 1
# kernel_size = 3, 3
# x = t.randn((b, ci, h, w))
# torch_output = t.max_pool2d(x, kernel_size, stride=stride, padding=padding,)
# my_output = maxpool2d(x, kernel_size, stride=stride, padding=padding,)
# t.testing.assert_close(my_output, torch_output)



def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
	'''Return a new tensor with padding applied to the edges.

	x: shape (batch, in_channels, width), dtype float32

	Return: shape (batch, in_channels, left + right + width)
	'''

	B, C, W = x.shape
	output = x.new_full(size=(B, C, left + W + right), fill_value=pad_value)
	output[..., left : left + W] = x
	# Note - you can't use `left:-right`, because `right` might be zero.
	return output



def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
	'''Return a new tensor with padding applied to the edges.

	x: shape (batch, in_channels, height, width), dtype float32

	Return: shape (batch, in_channels, top + height + bottom, left + width + right)
	'''

	B, C, H, W = x.shape
	output = x.new_full(size=(B, C, top + H + bottom, left + W + right), fill_value=pad_value)
	output[..., top : top + H, left : left + W] = x
	return output



def conv1d(
	x: Float[Tensor, "b ic w"],
	weights: Float[Tensor, "oc ic kw"],
	stride: int = 1,
	padding: int = 0
) -> Float[Tensor, "b oc ow"]:
	'''
	Like torch's conv1d using bias=False.

	x: shape (batch, in_channels, width)
	weights: shape (out_channels, in_channels, kernel_width)

	Returns: shape (batch, out_channels, output_width)
	'''

	x_padded = pad1d(x, left=padding, right=padding, pad_value=0)

	b, ic, w = x_padded.shape
	oc, ic2, kw = weights.shape
	assert ic == ic2, "in_channels for x and weights don't match up"
	ow = 1 + (w - kw) // stride
	# note, we assume padding is zero in the formula here, because we're working with input which has already been padded

	s_b, s_ic, s_w = x_padded.stride()

	# Get strided x (the new height/width dims have the same stride as the original height/width-strides of x,
	# scaled by the stride (because we're "skipping over" x as we slide the kernel over it))
	# See diagram in hints for more explanation.
	x_new_shape = (b, ic, ow, kw)
	x_new_stride = (s_b, s_ic, s_w * stride, s_w)
	x_strided = x_padded.as_strided(size=x_new_shape, stride=x_new_stride)

	return einops.einsum(x_strided, weights, "b ic ow kw, oc ic kw -> b oc ow")



IntOrPair = Union[int, Tuple[int, int]]
Pair = Tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
	'''Convert v to a pair of int, if it isn't already.'''
	if isinstance(v, tuple):
		if len(v) != 2:
			raise ValueError(v)
		return (int(v[0]), int(v[1]))
	elif isinstance(v, int):
		return (v, v)
	raise ValueError(v)

# Examples of how this function can be used:



def conv2d(
	x: Float[Tensor, "b ic h w"],
	weights: Float[Tensor, "oc ic kh kw"],
	stride: IntOrPair = 1,
	padding: IntOrPair = 0
) -> Float[Tensor, "b oc oh ow"]:
	'''
	Like torch's conv2d using bias=False

	x: shape (batch, in_channels, height, width)
	weights: shape (out_channels, in_channels, kernel_height, kernel_width)

	Returns: shape (batch, out_channels, output_height, output_width)
	'''

	stride_h, stride_w = force_pair(stride)
	padding_h, padding_w = force_pair(padding)

	x_padded = pad2d(x, left=padding_w, right=padding_w, top=padding_h, bottom=padding_h, pad_value=0)

	b, ic, h, w = x_padded.shape
	oc, ic2, kh, kw = weights.shape
	assert ic == ic2, "in_channels for x and weights don't match up"
	ow = 1 + (w - kw) // stride_w
	oh = 1 + (h - kh) // stride_h

	s_b, s_ic, s_h, s_w = x_padded.stride()

	# Get strided x (new height/width dims have same stride as original height/width-strides of x, scaled by stride)
	x_new_shape = (b, ic, oh, ow, kh, kw)
	x_new_stride = (s_b, s_ic, s_h * stride_h, s_w * stride_w, s_h, s_w)
	x_strided = x_padded.as_strided(size=x_new_shape, stride=x_new_stride)

	return einops.einsum(x_strided, weights, "b ic oh ow kh kw, oc ic kh kw -> b oc oh ow")

def maxpool2d(
	x: Float[Tensor, "b ic h w"],
	kernel_size: IntOrPair,
	stride: Optional[IntOrPair] = None,
	padding: IntOrPair = 0
) -> Float[Tensor, "b ic oh ow"]:
	'''
	Like PyTorch's maxpool2d.

	x: shape (batch, channels, height, width)
	stride: if None, should be equal to the kernel size

	Return: (batch, channels, output_height, output_width)
	'''

	# Set actual values for stride and padding, using force_pair function
	if stride is None:
		stride = kernel_size
	stride_h, stride_w = force_pair(stride)
	padding_h, padding_w = force_pair(padding)
	kh, kw = force_pair(kernel_size)

	# Get padded version of x
	x_padded = pad2d(x, left=padding_w, right=padding_w, top=padding_h, bottom=padding_h, pad_value=-t.inf)

	# Calculate output height and width for x
	b, ic, h, w = x_padded.shape
	ow = 1 + (w - kw) // stride_w
	oh = 1 + (h - kh) // stride_h

	# Get strided x
	s_b, s_c, s_h, s_w = x_padded.stride()

	x_new_shape = (b, ic, oh, ow, kh, kw)
	x_new_stride = (s_b, s_c, s_h * stride_h, s_w * stride_w, s_h, s_w)
	x_strided = x_padded.as_strided(size=x_new_shape, stride=x_new_stride)

	# Argmax over dimensions of the maxpool kernel
	# (note these are the same dims that we multiply over in 2D convolutions)
	output = t.amax(x_strided, dim=(-1, -2))
	return output


