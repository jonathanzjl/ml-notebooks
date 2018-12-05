#! /usr/bin/env python3
## ---------------------------------------------------------------------------
## Copyright (c) 2018 by General Electric Medical Systems
##
## vae_tf.py for VAE
##
## Made by Zhijin Li
## Mail:   <jonathan.li@ge.com>
##
## Started on  Mon Apr  9 17:43:51 2018 Zhijin Li
## Last update Tue Jul 17 11:30:37 2018 Zhijin Li
## ---------------------------------------------------------------------------


import math
import numpy as np
import tensorflow as tf


class VariationalAutoEncoder():
  """

  Class implementing the variational autoencoder (VAE).
  [reference](https://arxiv.org/abs/1312.6114).

  """

  def __init__(self,
               hidden_layer_type='fc',
               layer_cfg=[500, 500],
               fc_activation=tf.nn.tanh,
               conv_activation=tf.nn.relu,
               latent_dim=2,
               decoder_output_distr='bernoulli',
               conv_strides=[1, 1, 1, 1],
               pool_strides=[1, 2, 2, 1],
               pool_ksize=[1, 2, 2, 1],
               pad='SAME'):
    """
    Constructor for VAE.

    Args:
    ----------
    layer_type: str
    Type specifying the hidden layers. Can be
    - 'fc', indicating fully connected layers. In
      this case the encoder & decoder will be
      multi-layer perceptrons.
    - 'conv', indicating convolutional layers. In
      this case the encoder & decoder will be
      convolutional neural nets.
    Defaults to 'fc'.

    layer_cfg: array-like
    an array indicating the config of hidden layers.
    The length of the array will be the number of
    hidden layers for encoder and decoder (i.e.
    the overall number of layers will be
    2*len(layer_cfg)).

    In case of fully-connected hidden layers, an
    array of integers is expected. Each element in
    the list specifies the number of hidden nodes
    in the corresponding hidden layer.

    In case of convolutional hidden layers, a
    list with 3 elements is expected. The first two
    elements indicate the spatial size (height, width)
    of te 2D conv filters and the third one indicates
    the number of filters.

    This layer_cfg does not take the latent space
    layers into account. For both fc and conv cases,
    two fc operations, at the end of encoder and
    in the beginning of the decoder, are performed
    to encode hidden layer output to latent variable
    and to decode latent variable to hidden layer
    input. These two fc operations are not configured
    by layer_cfg.

    Defaults to [500, 500].

    fc_activation: tf.keras.activations
    Activation function in case where hidden layers
    are fc. Defaults to 'tanh'.

    conv_activation: tf.keras.activations
    Activation function in case where hidden layers
    are conv. Defaults to 'relu'.

    latent_dim: int
    Dimension of the latent variable. Defaults to 2.

    decoder_output_distr: str
    String indicating the distribution type of
    the decoder output. Can be 'gaussian', in
    case of gray level images or 'bernoulli' in
    case of binary images. Defaults to 'bernoulli'.

    conv_strides: list(int)
    4 integers specifying strides for 2D conv and
    2D transposed convolutions (tconv). Defaults
    to unit-strides.

    pool_strides: list(int)
    4 integers specifying 2D MaxPooling strides.
    Defaults to [1,2,2,1].

    pool_ksize: list(int)
    4 integers specifying 2D MaxPooling kernel
    sizes. Defaults to [1, 2, 2, 1], i.e. 2 x 2
    MaxPooling.

    pad: str
    Padding scheme for all convolutions, poolings
    and transposed convolutions. Defaults to 'SAME'.
    Using 'SAME' padding is a convenient way to
    ensure a match between sizes of conv+pool and
    tconv, for arbitrary kernel size and strides.

    """
    self.hidden_layer_type = hidden_layer_type
    self.decoder_output_distr = decoder_output_distr
    self.__check_params()

    self.layer_cfg = layer_cfg
    self.fc_activation = fc_activation
    self.conv_activation = conv_activation
    self.latent_dim = latent_dim

    self.conv_strides = conv_strides
    self.pool_strides = pool_strides
    self.pool_ksize   = pool_ksize
    self.pad          = pad
    self.small        = 1e-3

  def init_network(self, im_shape, optimizer, log_dir):
    """
    Initialize network architecture.

    Args:
    ----------
    im_shape: array-like
    Shape of each input image. For an autoencoder
    with fully-connected layers, images are assumed
    to be flattened. In this case, `im_shape` is a
    list with a single number n_elem, where n_elem
    is the flattened image size. For an autoencoder
    with convolutional layers, `im_shape` has the
    following three elements:
        (height, width, n_channels).

    optimizer: tf.keras.optimizers
    A tensorflow optimizer instance.

    log_dir: str
    Directory for log writing.

    """
    self.__init_tensors(im_shape)
    self.__init_architecture()
    self.__init_optimization(optimizer)

    self.__init = tf.global_variables_initializer()
    self.sess = tf.Session()
    self.sess.run(self.__init)

    self.__make_logger(log_dir)
    self.summaries = tf.summary.merge_all()


  def __init_tensors(self, im_shape):
    """
    Initialize all Tensors and shapes.

    Args:
    ----------
    im_shape: array-like
    Shape of each input image. For fully
    connected autoencoder, the image is
    assumed to be flattened. In this case,
    `im_shape` is a list with a single number
    n_elem, where n_elem is the flattened
    image size. For convolutional autoencoder
    `im_shape` has 3 elements:
        (height, width, n_channels).

    """
    self.__init_tensor_register()
    self.__init_input(im_shape)


  def __init_architecture(self):
    """
    Initialization encoder, latent space
    and decoder.

    """
    self.build_encoder()
    self.sample_latent()
    self.build_decoder()


  def __init_optimization(self, optimizer):
    """
    Initialize loss and optimizer.

    Args:
    ----------
    optimizer: tf.keras.optimizers
    A tensorflow optimizer instance.

    """
    self.build_loss()
    self.optimizer = optimizer
    self.train_op = self.optimizer.minimize(
      self.loss, name='optimization')


  def __check_params(self):
    """
    Check the format for input parameters
    hidden_layer_type and decoder_output_distr.

    Raise exception if format is incorrect.
    See class constructor regarding correct
    format.

    """
    self.__check_hidden_type()
    self.__check_output_distr()


  def __check_hidden_type(self):
    """
    Check the format for input parameter
    hidden_layer_type.

    Raise exception if format is incorrect.
    See class constructor regarding correct
    format.

    """
    if self.hidden_layer_type not in ['fc', 'conv']:
      raise Exception(
        'hidden type neither fc nor conv')


  def __check_output_distr(self):
    """
    Check the format for input parameter
    decoder_output_distr.

    Raise exception if format is incorrect.
    See class constructor regarding correct
    format.

    """
    if self.decoder_output_distr not in [
        'gaussian', 'bernoulli']:
      raise Exception(
        'output neither gaussian nor bernoulli.')


  def __make_logger(self, log_dir):
    """
    Create a logger for tensorboard.

    The logger is a tf.summary.FileWriter that
    writes training summary information to a
    specific directory.

    Args:
    ----------
    log_dir: str
    Directory for log writing.

    """
    self.logger = tf.summary.FileWriter(
      logdir=log_dir, graph=self.sess.graph)


  def summarize_hist(self, tensor_list):
    """
    Summarize a list of tensors and log their
    histograms.

    Args:
    ----------
    tensor_list: array-like(Tensor)
    An array-like of Tensors to summarize.

    """
    # Seems to be a bug in Tensorflow 1.4.1
    # related to summary variable names.
    # Added replace ':'->'_' to avoid warning
    # messages.
    for tensor in tensor_list:
      tf.summary.histogram(tensor.name.replace(
        ':','_'), tensor)


  def summarize_scalar(self, scalar_list):
    """
    Summarize and log a lis of scalar values, such
    as the loss value. A scalar value is a Tensor
    with a single value.

    Args:
    ----------
    scalar_list: a list of scalar
    A list of scalar values to summarize.

    """
    for scalar in scalar_list:
      tf.summary.scalar(scalar.name.replace(
        ':','_'), scalar)


  def __write_summaries(self, summaries, epoch):
    """
    Write summary information to disk.

    Args:
    ----------
    summaries: str
    Serialized summary protobuffer.

    """
    self.logger.add_summary(summaries, epoch)
    self.logger.flush()


  def __make_op_name(self, branch, tag):
    """
    Create a name for the operation given
    branch and output Tensor tag.

    Args:
    ----------
    branch: str
    Name of the branch. Can be 'encoder',
    'latent' or 'decoder' etc.

    tag: convertible to str
    Tag for the output Tensor. Can be an
    index or a name.

    Returns:
    ----------
    A string name for the operation.

    """
    return '{}_{}'.format(branch, tag)


  def __get_tensor(self, branch, tag):
    """
    Get output Tensor from specific branch
    and indx from the internal Tensor dict.

    Args:
    ----------
    branch: str
    Name of the branch.

    tag: convertible to str
    Tag for the output Tensor. Can be an
    index or a name.

    Returns:
    ----------
    The corresponding Tensor.

    """
    return self.tensors[
      self.__make_op_name(branch, tag)]


  def __init_tensor_register(self):
    """
    Initialize the tensor register.

    This creates an internal dictionary to
    register tensors create in all layers
    of VAE.

    """
    self.tensors = dict()


  def __init_input(self, im_shape):
    """
    Initialize the input Tensor with specific
    shape.

    The input Tensor is a tensorflow placeholder
    for input data feeding.

    If hidden layers are fully-connected,
    it is assumed that input images are
    flattened. In this case, the `placeholder`
    will have shape:
          (None, n_elems)
    where n_elems is the number of elements
    in the flattened image, i.e.:
          height x width x n_channels.

    If hidden layers are convolutional, the
    `placeholder` will have shape:
          (None, height, width, n_channels)

    The batch_size is always set to be None,
    to allow for forward pass with variable
    number of images.

    This function also register the input
    in the internal tensor register.

    Args:
    ----------
    im_shape: array-like
    Shape of the image. It corresponds to the
    shape of the placeholder excluding the
    batch_size.

    """
    op_name = 'input'
    with tf.variable_scope(op_name):
      self.input_tensor = tf.placeholder(
        dtype=tf.float32, shape=(None,*im_shape))
    self.tensors[op_name] = self.input_tensor
    self.__inshape = self.input_tensor.get_shape().as_list()


  def __init_encoder_params_fc(self):
    """
    Initialize the encoder layer parameter list
    when the hidden encoder and decoder layers
    are fully-connected.

    """
    self.enc_params_fc = self.layer_cfg


  def __init_encoder_params_conv(self):
    """
    Initialize the encoder layer parameters
    when the hidden encoder and decoder layers
    are convolutional.

    This function creates a numpy array with
    shape:
        len(self.layer_cfg) x 4.
    and with values parsed from self.layer_cfg.

    Each row represents the filter shape params
    of the corresponding encoder convolutional
    hidden layer.

    The four elements in each row indicates:
        filter_size_x, filter_size_y, n_channels, n_filters.

    Each row can be passed to __make_conv_wb to
    create a bank of convolutional filters and
    biases.

    """
    kern_size = [lay[:2] for lay in self.layer_cfg]
    n_filters = [[lay[-1]] for lay in self.layer_cfg]
    n_channel = [[self.__inshape[-1]]] + n_filters[:-1]
    self.enc_params_conv = np.concatenate(
      (kern_size, n_channel, n_filters), axis=1)


  def __init_decoder_params_fc(self):
    """
    Initialize the decoder layer parameter list
    when the hidden encoder and decoder layers
    are fully-connected.

    Each element represents the fan_out param of
    corresponding fully-connected decoder hidden
    layer.

    """
    self.dec_params_fc = list(reversed(
      [self.__inshape[-1]]+self.layer_cfg[:-1]))


  def __init_decoder_params_tconv(self):
    """
    Initialize the encoder transposed conv (tconv)
    layer parameters when the hidden encoder and
    decoder layers are convolutional.

    This function creates a list of tuples. Each
    element in the list can be used to configure
    the corresponding decoder hidden layer.

    Consider an arbitrary tuple in the list. The
    tuple itself contains two lists.

    The first list in the tuple has 4 elements
    representing the filter shape params of the
    corresponding decoder tconv hidden layer:
        ksize_w, ksize_h, n_filters, n_in_channels.

    Notice that the order of n_filters and
    n_in_channels is reversed compared to the shape
    Tensor created for a conv layer. This is
    because tconv and conv are symmetric operations.
    Tensorflow's tf.nn.conv2d_transposed and tf.nn.conv2d
    APIs use the same shape for the weights Tensor.
    However the meaning of the last two dimensions
    of the weight Tensor is reversed.

    These elements can be passed to __make_conv_wb
    to create a bank of transposed convolutional
    filters and biases.

    The second list in the tuple has 3 elements,
    representing the output_shape param of the
    corresponding decoder transposed convolutional
    hidden layer:
        height, width, n_channels.

    Notice that the actual value passed to
    tf.nn.conv2d_transpose's `output_shape` argument
    is a 4D Tensor that also contains the `n_batch`
    parameter, which is not considered in the list.
    The `n_batch` parameter can be queried from
    tf.nn.conv2d_transpose's inbound Tensor by
    calling tf.shape() and taking the first shape
    element.

    Notice that the output_shape is mandatory
    for tf.nn.conv2d_transpose API, since the
    shape of the output Tensor may not be unique
    give the shape of the input Tensor shape and
    the filters, and thus cannot be deduced
    automatically.

    """
    filters = np.flipud(self.enc_params_conv)
    output_shapes = np.array(list(reversed(
      [ self.tensors['input'].get_shape().as_list()[1:] ]+
      [ self.__get_tensor('encoder',id).get_shape().
        as_list()[1:] for id in range(1,len(self.layer_cfg))])))
    self.dec_params_tconv = list(zip(filters, output_shapes))


  def __init_latent_decode_params(self):
    """
    Initialize parameters for decoding latent
    samples.

    Two parameters are considered:
    - The decoded latent flatsize, which is the
      size of the flattened decoded latent sample.
    - The decoded latent spatial size, which is
      the spatial size of the latent sample when
      it is reshaped as an image. It is equal to
      the (height, width, n_channels) of the last
      Tensor output from the last encoder hidden
      layer. Notice that `n_batch` is not considered,
      it can be queried from other Tensors by
      calling tf.shape() and taking the first element.

    """
    ref_tensor = self.__get_tensor('encoder',len(self.layer_cfg))
    self.latent_flatsize = np.prod(
      ref_tensor.get_shape().as_list()[1:])
    if self.hidden_layer_type == 'conv':
      self.latent_spatsize = ref_tensor.get_shape().as_list()[1:]


  def __make_weights(self, shape, op_name):
    """
    Create weight Tensors for fully connected (fc),
    convolution (conv) or transposed convolution
    (tconv) operations. For conv and tconv ops
    the weight Tensors are also referred to as
    filters.

    Args:
    ----------

    shape: array-like
    Shape parameters of the weight Tensor.

    For fc op, it contains two elements:
        fan_in, fan_out
    For conv and tconv ops, it contains four
    elements. In case of conv op: filter_height,
    filter_width, n_channels and n_filters. In
    case of tconv op: filter_height, filter_width,
    n_filters and n_channels.

    Notice that the position of n_filters and
    n_channels are reversed for conv and tconv.
    This is intentional.

    op_name: str
    Name of the operation. Important to make
    Tensor's name unique in scope.

    Returns:
    ----------
    A weight Tensor (filter).

    """
    weights = tf.get_variable(
      name='{}_weights'.format(op_name),
      shape=shape,
      dtype=tf.float32,
      initializer=tf.glorot_normal_initializer())
    # self.summarize_hist([weights])
    return weights

  def __make_biases(self, n_neurons, op_name):
    """
    Create biases Tensor.

    Args:
    ----------

    n_neurons: int
    Size of bias Tensor. For conv or tconv ops
    this corresponds to `n_filters`. For fc op
    this corresponds to the `fan_out` after
    matrix multiplication.

    op_name: str
    Name of the operation. Important to make
    Tensor's name unique in scope.

    Returns:
    ----------
    A bias Tensor.

    """
    biases = tf.get_variable(
      name='{}_biases'.format(op_name),
      shape=[n_neurons],
      dtype=tf.float32,
      initializer=tf.zeros_initializer())
    # self.summarize_hist([biases])
    return biases

  def __make_fc_wb(self, fan_in, fan_out, op_name):
    """
    Create weights and biases for a fully-
    connected (fc) operation.

    Args:
    ----------
    fan_in: int
    Size of input Tensor for the fc operation.

    fan_out: int
    Size of output Tensor for the fc operation.

    op_name: str
    Name of the operation. Important to make
    Tensor's name unique in scope.

    Returns:
    ----------
    A tuple of two Tensors for weights and
    biases.

    """
    weights = self.__make_weights((fan_in,fan_out),op_name)
    biases  = self.__make_biases(fan_out, op_name)
    return (weights, biases)


  def __make_conv_wb(self, params, op_name):
    """
    Create weights (filters) and biases for a
    convolution (conv) operations.

    Args:
    ----------

    params: array-like
    Params of the conv operation. It contains
    four elements: filter_height, filter_width
    n_channels and n_filters.

    op_name: str
    Name of the operation. Important to make
    Tensor's name unique in scope.

    Returns:
    ----------
    A tuple of two Tensors for weights (filters)
    and biases.

    """
    filters = self.__make_weights(params,op_name)
    biases  = self.__make_biases(params[-1],op_name)
    return (filters, biases)


  def __make_tconv_wb(self, params, op_name):
    """
    Create weights (filters) and biases for a
    transposed convolution (tconv) operations.

    Args:
    ----------

    params: array-like
    Params of the tconv operation. It contains
    four elements: filter_height, filter_width
    n_channels and n_filters.

    Notice that the position of n_filters and
    n_channels are inverted compared to params
    of a conv operation. This is intentional.

    op_name: str
    Name of the operation. Important to make
    Tensor's name unique in scope.

    Returns:
    ----------
    A tuple of two Tensors for weights (filters)
    and biases.

    """
    filters = self.__make_weights(params,op_name)
    biases  = self.__make_biases(params[2],op_name)
    return (filters, biases)


  def __apply_fc(self, tensor_in, fan_out,
                 activation, op_name):
    """
    Apply fully-connected operation, i.e. multiply
    an input Tensor by a weight matrix, add biases
    and apply activation function.

    Args:
    ----------
    tensor_in: Tensor
    The input Tensor.

    fan_out: int
    Size of the output Tensor.

    activation: function
    A tensorflow activation function.

    op_name: str
    Name of the operation.

    Returns:
    ----------
    The output Tensor.

    """
    fan_in = tensor_in.get_shape().as_list()[-1]
    weights, biases = self.__make_fc_wb(fan_in,fan_out,op_name)
    tensor_out = activation(tf.add(
      tf.matmul(tensor_in,weights),biases),name=op_name)
    return tensor_out


  def __apply_conv_pool(self, tensor_in, params,
                        activation, op_name):
    """
    Apply convolution and max pooling operations.

    Convolve the input Tensor using a set of
    filters, add biases, apply activation and
    do max pooling.

    Args:
    ----------
    tensor_in: Tensor
    The input Tensor.

    params: array-like
    An array-like with 4 elements: filter_height,
    filter_width, n_channels and n_filters.

    activation: function
    A tensorflow activation function.

    op_name: str
    Name of the operation.

    Returns:
    ----------
    The output Tensor.

    """
    weights, biases = self.__make_conv_wb(params,op_name)
    tensor_out = tf.nn.max_pool(
      activation(tf.nn.conv2d(
        tensor_in, weights, strides=self.conv_strides,
        padding=self.pad) + biases), ksize=self.pool_ksize,
      strides=self.pool_strides, padding=self.pad,
      name=op_name)
    return tensor_out


  def __apply_tconv(self, tensor_in, params,
                    activation, op_name):
    """
    Apply transposed convolution (tconv) operation.

    Tconv operations are performed only in
    the decoder branch. It upsamples a feature map
    to its original size before convolution and
    pooling at the symmetric layer in the encoder
    branch.

    Args:
    ----------
    tensor_in: Tensor
    The input Tensor.

    params: tuple(list)
    A tuple of two lists for params of the tconv operation.
    The first list contains filter_height, filter_width,
    n_filters and n_in_channels. It is used to create
    filter and bias Tensors for the tconv operation. The
    second list contains height, width and n_out_channels,
    which correspond to the height, width and n_channels
    of the Tensor at the symmetric layer in the encoder
    branch. The second list is needed to allow the tconv
    operation to produce Tensor with desired size, which
    cannot be deduced automatically. It is further
    concatenated with `n_batch` queried from the input
    tensor, and passed to the `output_shape` argument of
    tf.nn.conv2d_transpose.

    activation: function
    A tensorflow activation function.

    op_name: str
    Name of the operation.

    Returns:
    ----------
    The output Tensor.

    """
    weights, biases = self.__make_tconv_wb(params[0],op_name)
    tensor_out = activation(
      tf.nn.conv2d_transpose(
        tensor_in, weights, strides=self.pool_strides,
        output_shape=(tf.shape(tensor_in)[0],*params[1]),
        padding=self.pad) + biases, name=op_name)
    return tensor_out


  def __make_hiddens(self, tensor_in, branch, layer_func,
                     layer_params, activation):
    """
    Build the graph for a series of hidden
    layers for encoder or decoder.

    The latent layers are not considered.

    Args:
    ----------
    tensor_in: Tensor
    The input Tensor.

    branch: str
    Name of the branch: 'encoder' or 'decoder'.

    layer_func: function
    Function for the hidden layers. Can be:
    - self.__apply_fc for fully connected
      hidden layers.
    - self.__apply_conv_pool for convolutional
      hidden layers.
    - self.__apply_tconv for transposed
      convolutional hidden layers in decoder branch.

    layer_params: Tensor
    Config of the layers.

    activation: function
    A tensorflow activation function.

    Returns:
    ----------
    The output Tensor of the last operation.

    """
    tensor = tensor_in
    for indx, params in enumerate(layer_params):
      op_name = self.__make_op_name(branch,indx+1)
      with tf.variable_scope(op_name):
        tensor = layer_func(
          tensor, params, activation, op_name)
        self.tensors[op_name] = tensor
    return tensor


  def build_encoder(self):
    """
    Build the encoder network.

    The encoder transforms each input batch
    to parameters of the latent distribution,
    assumed to be Gaussian. These parameters
    are latent mean and latent log(sigma^2).

    """
    if self.hidden_layer_type == 'fc':
      tensor = self.__build_encoder_fc()
    elif self.hidden_layer_type == 'conv':
      tensor = self.__build_encoder_conv()
    with tf.variable_scope('latent_space'):
      self.__encode_latent(tensor)


  def __build_encoder_fc(self):
    """
    Build the encoder network with fc hidden
    layers.

    Returns:
    ----------
    The output Tensor.

    """
    self.__init_encoder_params_fc()
    return self.__make_hiddens(
      self.input_tensor, 'encoder', self.__apply_fc,
      self.enc_params_fc, self.fc_activation)


  def __build_encoder_conv(self):
    """
    Build the encoder network with conv hidden
    layers.

    Returns:
    ----------
    The output Tensor. It is flattened to be
    compatible with latent fc layer.

    """
    self.__init_encoder_params_conv()
    tensor = self.__make_hiddens(
      self.input_tensor,'encoder',self.__apply_conv_pool,
      self.enc_params_conv, self.conv_activation)
    return self.__flatten(tensor)


  def __encode_latent(self, tensor_in):
    """
    Encode the output Tensor of the last
    hidden layer in encoder to parameters
    of latent space distribution.

    The latent space distribution is assumed
    to be multi-variate Gaussian with mean =
    `latent_mean` and standard deviation sigma.
    Note that VAE does not directly output
    the sigma, but outputs log(sigma^2) instead.
    See: [the original VAE paper](https://arxiv.org/abs/1312.6114)
    for more details.

    The encoding is done by applying a fc
    operation to the (flattened) input Tensor
    with fan_out equal to the latent space
    dimension.

    Args:
    ----------
    tensor_in: Tensor.
    The input Tensor. Assumed to be flattened.

    """
    with tf.variable_scope('latent_distr'):
      self.latent_mean = self.__apply_fc(
        tensor_in,self.latent_dim,tf.identity,'distr_mean')
      self.latent_stdv = self.__apply_fc(
        tensor_in,1,tf.nn.sigmoid,'distr_stdv')
      # self.summarize_hist(
      #   [self.latent_mean, self.latent_lsg2])


  def sample_latent(self):
    """
    Draw samples from the latent distribution.

    Draw random variables from N(0,1), then
    transform them to the follow latent distribution,
    i.e. N(latent_mean, latent_stdv).

    According to the [original VAE paper](https://arxiv.org/abs/1312.6114),
    for large batch size (typically >= 100),
    drawing one Tensor sample is enough for a
    reliable estimation of the expected log-likelihood.

    Returns:
    ----------
    Sampled Tensor with variables following
    latent distribution. Shape of the Tensor
    is `n_bacth x latent_dim`.

    """
    with tf.variable_scope('latent_samples'):
      eps = tf.random_normal(
        (tf.shape(self.latent_mean)[0],
         self.latent_dim), mean=0, stddev=1,
        dtype=tf.float32, name='auxiliary')
      self.latent_samples = tf.add(
        tf.multiply(eps, self.latent_stdv),
        self.latent_mean, name='samples')
        # tf.multiply(eps, tf.exp(self.latent_lsg2/2.)),


  def __decode_latent(self, latent_samples):
    """
    Decode a latent sample.

    The decoding is done by applying a fc
    operation with fan_out equal to the number
    of neurons in the last encoder layer.

    Args:
    ----------
    latent_samples: Tensor.
    The input latent samples with the following
    shape:
        n_batch x latent_dim

    Returns:
    ----------
    The decoded Tensor. It is flattened with
    shape:
        n_batch, n_neurons_out
    where n_neurons_out is equal to the number
    of the neurons in the flattened Tensor of
    the last encoder layer.

    """
    with tf.variable_scope('latent_decoded'):
      tensor = self.__apply_fc(
        latent_samples, self.latent_flatsize,
        self.fc_activation, 'latent_decoded')
      return tensor


  def __flatten(self, tensor_in):
    """
    Flatten a Tensor returned by a convolution
    operation in the encoder.

    Suppose tensor_in has shape:
        `(n_batch, height, width, n_channels)`
    The function returns a tensor with shape:
        `(n_batch, height*width*n_channels)`

    Args:
    ----------
    tensor_in: Tensor
    The input Tensor to flatten.

    register: bool
    Whether to register the Tensor in internal
    Tensor dictionary. Defaults to True.

    Returns:
    ----------
    The flattened Tensor. See explanation above.

    """
    flat_size = np.prod(
      tensor_in.get_shape().as_list()[1:])
    tensor = tf.reshape(tensor_in, [-1, flat_size])
    return tensor


  def __reshape_decoded_latent(self, tensor_in):
    """
    Reshape the flattened Tensor output from the
    fc layer performed on latent samples. This
    aims to prepare the input with proper size
    for follow-up transposed conv decoder.

    Suppose tensor_in has shape:
        `(n_batch, height*width*n_channels)`
    The function returns a tensor with shape:
        `(n_batch, height, width, n_channels)`

    Here height, width, n_channels are queried
    from the output shape of the last convolution
    layer in the encoder.

    Args:
    ----------
    tensor_in: Tensor
    The input flattened Tensor to reshape.

    Returns:
    ----------
    The reshaped Tensor. See explanation above.

    """
    with tf.variable_scope('reshape'):
      shape = (tf.shape(tensor_in)[0],*self.latent_spatsize)
      tensor = tf.reshape(
        tensor_in, shape=shape, name='latent_reshaped')
      return tensor


  def build_decoder(self):
    """
    Build the decoder network.

    The decoder transforms latent samples
    to params of the decoder output distr,
    which can be Gaussian or Bernoulli.

    """
    self.__init_latent_decode_params()
    tensor = self.__decode_latent(self.latent_samples)
    if self.hidden_layer_type == 'fc':
      self.__build_decoder_fc(tensor)
    elif self.hidden_layer_type == 'conv':
      tensor = self.__reshape_decoded_latent(tensor)
      self.__build_decoder_tconv(tensor)


  def __build_decoder_fc(self, tensor):
    """
    Build the decoder network with fc hidden
    layers.

    Args:
    ----------
    The input Tensor.

    Returns:
    ----------
    The output Tensor.

    """
    self.__init_decoder_params_fc()
    tensor = self.__make_hiddens(
      tensor, 'decoder', self.__apply_fc,
      self.dec_params_fc[:-1], self.fc_activation)
    self.__make_output_fc(tensor)


  def __build_decoder_tconv(self, tensor):
    """
    Build the encoder network with tconv hidden
    layers.

    Args:
    ----------
    The input Tensor.

    Returns:
    ----------
    The output Tensor.

    """
    self.__init_decoder_params_tconv()
    tensor = self.__make_hiddens(
      tensor, 'decoder', self.__apply_tconv,
      self.dec_params_tconv[:-1],self.conv_activation)
    self.__make_output_tconv(tensor)


  def __make_output_fc(self, tensor):
    """
    Construct the decoder output Tensors for fc
    hidden layers.

    For Bernoulli output distr, the output mean
    will be computed. For Gaussian output distr
    an extra standard deviation (sigma) will be
    computed.

    Args:
    ----------
    The input Tensor.

    """
    with tf.variable_scope('decoder_out'):
      self.output_mean = self.__apply_fc(
        tensor,self.dec_params_fc[-1],tf.nn.sigmoid,'out_mean')
      if self.decoder_output_distr == 'gaussian':
        self.output_stdv = self.__apply_fc(
          tensor,1,tf.nn.softplus,'out_stdv')
      #   self.summarize_hist([self.output_stdv])
      # self.summarize_hist([self.output_mean])


  def __make_output_tconv(self, tensor):
    """
    Construct the decoder output Tensors for tconv
    hidden layers.

    For Bernoulli output distr, the output mean
    will be computed. For Gaussian output distr
    an extra standard deviation (sigma) will be
    computed.

    Args:
    ----------
    The input Tensor.

    """
    with tf.variable_scope('decoder_out'):
      self.output_mean = self.__apply_tconv(
        tensor,self.dec_params_tconv[-1],tf.nn.sigmoid,'out_mean')
      if self.decoder_output_distr == 'gaussian':
        self.output_stdv = self.__apply_fc(
          self.__flatten(tensor),1,tf.nn.softplus,'out_stdv')
      #   self.summarize_hist([self.output_stdv])
      # self.summarize_hist([self.output_mean])


  def build_loss(self):
    """
    Build the VAE loss function.

    The loss of VAE is the sum of two terms:
    the reconstruction loss and the latent
    regularization.

    """
    with tf.variable_scope('loss'):
      latent_reg = tf.reduce_mean(
        self.__make_latent_reg(), name='latent_reg_loss')
      if self.decoder_output_distr == 'gaussian':
        recon_loss = tf.reduce_mean(
          self.__recon_loss_gauss(), name='recon_loss')
      elif self.decoder_output_distr == 'bernoulli':
        recon_loss = tf.reduce_mean(
          self.__recon_loss_berno(), name='recon_loss')
      self.loss = tf.add(
        latent_reg, recon_loss, name='vae_loss')
      self.summarize_scalar(
        [latent_reg, recon_loss, self.loss])


  def __make_latent_reg(self):
    """
    Compute latent space regularization term
    in the VAE loss function.

    This is the negative KL-divergence from
    the approximate posterior of the latent
    distribution (Gaussian)

    See https://arxiv.org/abs/1312.6114 for
    detail.

    Returns:
    ----------
    A Tensor with computed latent
    regularization.

    """
    # latent_reg = self.__flatten(
    #   tf.square(self.latent_mean)+tf.exp(self.latent_lsg2)
    #   - self.latent_lsg2 - 1)
    latent_reg = self.__flatten(
      tf.square(self.latent_mean)+tf.square(self.latent_stdv)
      - tf.log(tf.square(self.latent_stdv)) - 1)
    return 0.5*tf.reduce_sum(latent_reg, axis=1)


  def __recon_loss_gauss(self):
    """
    Compute reconstruction loss term in the
    VAE loss function, for Gaussian case.

    This is equalvalent to negative log-likelihood
    of x conditioned on latent sample z.

    See https://arxiv.org/abs/1312.6114 for
    detail.

    Returns:
    ----------
    A Tensor with computed reconstruction
    loss.

    """
    squrd_diff = self.__flatten(tf.abs(
      self.input_tensor-self.output_mean))
    # n_features = tf.cast(tf.shape(squrd_diff)[1],tf.float32)
    # recon_loss = squrd_diff/(2.*tf.square(
    #   self.output_stdv+self.small))+0.5*n_features*tf.log(
    #     2*math.pi)+n_features*tf.log(self.output_stdv+self.small)
    # return tf.reduce_sum(recon_loss, axis=1)
    return tf.reduce_sum(squrd_diff, axis=1)


  def __recon_loss_berno(self):
    """
    Compute reconstruction loss term in the
    VAE loss function, for Bernoulli case.

    This is equalvalent to negative log-likelihood
    of x conditioned on latent sample z.

    See https://arxiv.org/abs/1312.6114 for
    detail.

    Returns:
    ----------
    A Tensor with computed reconstruction
    loss.

    """
    recon_loss = self.__flatten(
      -self.input_tensor*tf.log(self.small+self.output_mean)
      -(1-self.input_tensor)*tf.log(
        self.small+1-self.output_mean))
    return tf.reduce_sum(recon_loss, axis=1)


  def __fit_batch(self, batch_in):
    """
    Train VAE for 1 epoch with a batch of
    input, and evaluate loss value.

    Args:
    ----------
    batch_in: Tensor
    The input batch.

    Returns:
    ----------
    A pair of values
    - The loss after 1 epoch of training.
    - The summary after 1 epoch of training.

    """
    __, summaries, loss_val = self.sess.run(
      [self.train_op, self.summaries, self.loss],
      feed_dict={self.input_tensor: batch_in})
    return (summaries, loss_val)


  def train(self, optimizer, input_shape, dataset_size,
            batch_gen, batch_size=128, n_epochs=100,
            log_dir='./log'):
    """
    Train VAE for a specific number of epochs.

    Args:
    ----------
    optimizer: optimizer instance
    A tensorflow optimizer instance.

    input_shape: array-like
    Shape of each input image. For an autoencoder
    with fully-connected layers, images are assumed
    to be flattened. In this case, `im_shape` is a
    list with a single number n_elem, where n_elem
    is the flattened image size. For an autoencoder
    with convolutional layers, `im_shape` has the
    following three elements:
        (height, width, n_channels).

    dataset_size: int
    Total number of training images in dataset.

    batch_gen: function or lambda
    A functional object that takes two arguments:
    1. the batch_size,
    2. the batch index, an integer between 0 and
       n_batch-1,
    and returns a batch of training data.

    batch_size: int
    Number of training samples for each batch.

    n_epochs: int
    Number of total training epochs.

    """
    self.init_network(input_shape,optimizer,log_dir)
    n_batches = int(dataset_size/float(batch_size))
    for epch in range(n_epochs):
      __avg = 0.
      for __b in range(n_batches):
        batch = batch_gen(batch_size, __b)
        summ, loss = self.__fit_batch(batch)
        __avg += loss/n_batches
      if summ is not None: self.__write_summaries(summ, epch)
      print('Epoch: {} - Loss: {}.'.format(epch,__avg))
    self.logger.close()


  def __sample_gauss_decoder(self, entry, data):
    """
    Sampling from Gaussian decoder.

    Args:
    ----------
    entry: Tensor
    The entry Tensor in the graph for data feeding.

    data: array
    Input data to feed to `entry`. For fc
    autoencoder, its shape is n_samples x n_elems.
    For conv autoencoder, its shape is:
        n_samples x height x width x n_channels.

    Returns:
    ----------
    An array of data samples decoded from `data`.
    Its shape follows the same rule as `data`.

    """
    mu, sigma = self.sess.run(
      [self.output_mean, self.output_stdv],
      feed_dict={entry: data})
    if self.hidden_layer_type == 'conv':
      sigma = np.reshape(
        sigma, newshape=(sigma.shape[0],1,1,sigma.shape[-1]))
    return np.random.randn(*mu.shape)*sigma+mu


  def __sample_berno_decoder(self, entry, data):
    """
    Sampling from Bernoulli decoder.

    Args:
    ----------
    entry: Tensor
    The entry Tensor in the graph for data feeding.

    data: array
    Input data to feed to `entry`. For fc
    autoencoder, its shape is n_samples x n_elems.
    For conv autoencoder, its shape is:
        n_samples x height x width x n_channels.

    Returns:
    ----------
    An array of data samples decoded from `data`.
    Its shape follows the same rule as `data`.

    """
    ps = self.sess.run(
      self.output_mean, feed_dict={entry: data})
    return np.random.binomial(1, ps, size=ps.shape)


  def __decode_from_feed(self, entry, data, use_sampling):
    """
    Get output from decoder when data is fed into
    specific entry Tensor.

    Args:
    ----------
    entry: Tensor
    The entry Tensor for data feeding.

    data: array
    Data samples fed to `entry`. For fc
    autoencoder, its shape is n_samples x n_elems.
    For conv autoencoder, its shape is:
        n_samples x height x width x n_channels.

    use_sampling: bool
    Whether to use sampling from decoder distr.
    If False, the decoder `output_mean` will be
    returned as the decoded data. If True, random
    samples following decoder distr will be drawn.

    Returns:
    ----------
    An array decoded output. Its shape follows the
    same rule as `data`.

    """
    if not use_sampling:
      return self.sess.run(
        self.output_mean, feed_dict={entry: data})
    else:
      if self.decoder_output_distr == 'gaussian':
        return self.__sample_gauss_decoder(entry,data)
      else:
        return self.__sample_berno_decoder(entry,data)


  def encode(self, data_samples):
    """
    Encode input data samples to latent variables.

    Args:
    ----------
    samples: array
    Input data samples. For fc autoencoder, the
    input shape is n_samples x n_elems. For conv
    autoencoder, the input shape is:
        n_samples x height x width x n_channels.

    Returns:
    ----------
    A tuple of latent variables corresponding
    to the input data samples. Each pair consists
    in the latent mean and standard deviation
    (sigma).

    """
    __mu, __lsg2 = self.sess.run(
      [self.latent_mean, self.latent_stdv],
      feed_dict={self.input_tensor: data_samples})
    return (__mu, np.exp(__lsg2/2.))


  def decode(self, latent_samples, use_sampling=False):
    """
    Decode latent samples to input data space.

    Args:
    ----------
    latent samples: array
    Input latent samples. For fc autoencoder, its
    shape is n_samples x n_elems. For conv
    autoencoder, its shape is:
        n_samples x height x width x n_channels.

    use_sampling: bool
    Whether to use sampling from decoder distr.
    If False, the decoder `output_mean` will be
    returned as the decoded data. If True, random
    samples following decoder distr will be drawn.

    Returns:
    ----------
    An array of decoded data samples. Its shape
    follows the same rule as the input latent
    samples.

    """
    return self.__decode_from_feed(
      self.latent_samples, latent_samples, use_sampling)


  def reconstruct(self, data_samples, use_sampling=False):
    """
    Reconstruct the input data.

    This is equivalent to running full forward pass
    of the encoder-decoder.

    Args:
    ----------
    samples: array
    Input data samples. For fc autoencoder, the
    input shape is n_samples x n_elems. For conv
    autoencoder, the input shape is:
        n_samples x height x width x n_channels.

    use_sampling: bool
    Whether to use sampling from decoder distr.
    If False, the decoder `output_mean` will be
    returned as the decoded data. If True, random
    samples following decoder distr will be drawn.

    Returns:
    ----------
    An array of reconstructed samples. Its shape
    follows the same rule as the input samples.

    """
    return self.__decode_from_feed(
      self.input_tensor, data_samples, use_sampling)
