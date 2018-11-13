from glob import glob

import tensorflow as tf
import numpy as np
import os
import scipy.misc as scm
#其中Ksize指卷积核的大小，outshape指输出的张量的shape，
# sted是一个bool类型的参数，表示用不同的方式初始化参数
def deconv2d(name, tensor, ksize, outshape,stride=2,sted=True,padding='SAME'):
    with tf.variable_scope(name):
        def uniform(stdev,size):
            return np.random.uniform(low=-stdev*np.sqrt(3),
                                     high=stdev*np.sqrt(3),
                                     size=size).astype('float32')
        tensor_dim = tensor.get_shape().as_list()
        in_dim = tensor_dim[3]
        out_dim = outshape[3]
        print(outshape)
        fan_in = in_dim*ksize**2/(stride**2)
        fan_out = np.array(out_dim*ksize**2)
        if sted:
            print('fan_out:')
            print(fan_out)
            print('fan_in:')
            print(fan_in)
            print('fan_out + fan_in:')
            print(fan_out + fan_in)
            filter_stdev = np.sqrt(4./(fan_out + fan_in))
        else:
            filter_stdev = 2/(in_dim*out_dim)
        print(filter_stdev)
        filter_value = uniform(filter_stdev,(ksize,ksize,out_dim,in_dim))

        w = tf.get_variable('weight',validate_shape=True,dtype=tf.float32,initializer=filter_value)

        var = tf.nn.conv2d_transpose(tensor,w,outshape,strides=[1,stride,stride,1],padding=padding)

        b = tf.get_variable('bias',[outshape[-1]],'float32',initializer=tf.constant_initializer(0.0))
        return tf.nn.bias_add(var,b)


def lrelu(x,leak=0.2):
    return tf.maximum(x,leak * x)


#全连接层fc的输入参数value指输入向量，output_shape指经过全连接层后输出的向量维度，
# 比如我们生成器这里噪声向量维度是128，我们输出的是4*4*8*64维
def fc(name,value,output_shape):
    with tf.variable_scope(name,reuse=True) as scope:
        shape = value.get_shape().as_list()
        def uniform(stdev,size):
            return np.random.uniform(low=-stdev*np.sqrt(3),
                                     high=stdev*np.sqrt(3),
                                     size=size).astype('float32')
        weight_val = uniform(np.sqrt(2./(shape[1]+output_shape)),(shape[1],output_shape))
        w = tf.get_variable('weight',validate_shape=True,dtype=tf.float32,
                            initializer=weight_val)
        b = tf.get_variable('bias',[output_shape],dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        return tf.matmul(value,w) + b

# 我们都希望权重都能初始化到一个比较好的数，所以这里我没有直接用固定方差的高斯分布去初始化权重，
# 而是根据每一层的输入输出通道数量的不同计算出一个合适的方差去做初始化。
# 同理，我们还封装了卷积操作，其形式如下：
def conv2d(name,tensor,ksize,out_dim,stride=2,sted=True,padding='SAME'):
    with tf.variable_scope(name):
        def uniform(stdev,size):
            return np.random.uniform(low=-stdev*np.sqrt(3),
                                     high=stdev*np.sqrt(3),
                                     size=size).astype('float32')
        tensor_dim = tensor.get_shape().as_list()
        fan_in = tensor_dim[3]*ksize**2
        fan_out = out_dim*ksize**2/(stride**2)
        if sted:
            filter_stdev = np.sqrt(4. / (fan_in + fan_out))
        else:
            filter_stdev = 2 / ( out_dim+tensor_dim[3])

        filter_value = uniform(filter_stdev,(ksize,ksize,tensor_dim[3],out_dim))

        w = tf.get_variable('weight',validate_shape=True,dtype=tf.float32,
                            initializer=filter_value)
        var = tf.nn.conv2d(tensor,w,[1,stride,stride,1],padding=padding)
        b = tf.get_variable('bias',[out_dim],'float32',initializer=tf.constant_initializer(0.0))

        return tf.nn.bias_add(var,b)

def load_data(data_dir,batch_size,idx):
    def get_image(img_path):
        img = scm.imread(img_path) / 255. - 0.5
        # img = img[...,::-1] rgb to bgr
        return img

    data = sorted(glob(os.path.join(data_dir, "*.*")))

    batch_files = data[idx * batch_size: (idx + 1) * batch_size]
    batch_data = [get_image(batch_file) for batch_file in batch_files]
    return batch_data
