import time

import plot as plot

from layer import *
import tensorflow as tf
import os
from image import save_img

class Train(object):
    def __init__(self,sess):
        self.sess = sess
        self.img_size = 64
        self.trainable = True
        self.batch_size = 64
        self.lr = 0.0002
        self.mm = 0.5
        self.z_dim = 128
        self.EPOCH = 50
        self.LAMBDA = 0.1
        self.model = 'WGAN_GP'
        self.dim = 3
        self.build_model()
    #生成器传进去三个参数，分别是名字， 输入数据，和一个bool型状态变量reuse，
    # 用来表示生成器是否复用，reuse=True代表网络复用，False代表不复用。
    # 生成器一共包括1个全连接层和4个转置卷积层，每一层后面都跟一个batchnorm层，
    # 激活函数都选择relu。其中fc(),deconv2d()函数和bn()函数都是我们封装好的函数，
    # 代表全链接层，转制卷积层，和归一化层，其形式如下：
    def generator(self,name,noise,reuse):
        with tf.variable_scope(name,reuse=reuse):
            output = fc('g_dc',noise,4*4*8*64)
            output = tf.reshape(output,[self.batch_size,4,4,8*64])
            output = tf.nn.relu(self.bn('g_bn1',output))
            print(self.img_size)
            output = deconv2d('g_dcon1',output,5,outshape=[self.batch_size,self.img_size // 8,self.img_size // 8,64*4])
            output = tf.nn.relu(self.bn('g_bn2',output))

            output = deconv2d('g_dcon2', output, 5, outshape=[self.batch_size, self.img_size // 4,  self.img_size // 4,64 * 2])
            output = tf.nn.relu(self.bn('g_bn3', output))

            output = deconv2d('g_dcon3', output, 5, outshape=[self.batch_size, self.img_size // 2, self.img_size // 2,64 * 1])
            output = tf.nn.relu(self.bn('g_bn4', output))

            output = deconv2d('g_dcon4', output, 5, outshape=[self.batch_size, self.img_size ,self.img_size , self.dim])

            return tf.nn.tanh(output)

    def bn(self,name,input):
        val = tf.contrib.layers.batch_norm(input,decay=0.9,
                                           updates_collections=None,
                                           epsilon=1e-5,
                                           scale=True,
                                           is_training=True,
                                           scope=name)
        return val


    def discriminater(self,name,input,reuse):
        with tf.variable_scope(name,reuse=reuse):
            output = conv2d('d_con1',input,5,64)
            output = lrelu(self.bn('d_bn1',output))
            output = conv2d('d_con2', output, 5, 64*2)
            output = lrelu(self.bn('d_bn2', output))
            output = conv2d('d_con3', output, 5, 64*4)
            output = lrelu(self.bn('d_bn3', output))
            output = conv2d('d_con4', output, 5, 64*8)
            output = lrelu(self.bn('d_bn4', output))
            output = tf.reshape(output,[self.batch_size,-1])
            output = fc('d_fc',output,1)
            return tf.nn.sigmoid(output),output


    def build_model(self):
        # build placeholders
        self.x = tf.placeholder(tf.float32,shape=[self.batch_size,self.img_size,self.img_size,self.dim],name='real_img')
        self.z = tf.placeholder(tf.float32,shape=[self.batch_size,self.z_dim],name='noise')
        #define the network
        self.G_img = self.generator('gen',self.z,reuse=False)
        self.D,self.D_logits = self.discriminater('dis',self.x,reuse=False)
        self.D_f,self.D_logits_f = self.discriminater('dis',self.G_img,reuse=True)

        #calculate the loss ,There are two model "DCGAN""WGAN-GP",the only one different of them is the loss function.
        if self.model == 'DCGAN':
            self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D),logits=self.D_logits))
            self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.D_f),logits=self.D_logits_f))
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_f),logits=self.D_logits_f))
            self.d_loss = self.d_loss_fake + self.d_loss_real
        elif self.model == 'WGAN_GP':
            self.g_loss = tf.reduce_mean(self.D_logits_f)
            disc_cost = -tf.reduce_mean(self.D_logits_f) + tf.reduce_mean(self.D_logits)

            alpha = tf.random_uniform(shape=[self.batch_size,1],minval=0,maxval=1.)
            differences = self.G_img - self.x
            interpolates = self.x + (alpha * differences)
            _,D_logits = self.discriminater('dis',interpolates,reuse=True)
            gradients = tf.gradients(D_logits,[interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            self.d_loss = disc_cost + self.LAMBDA * gradient_penalty

        all_vars = tf.global_variables()
        g_vars = [v for v in all_vars if 'gen' in v.name]
        d_vars = [v for v in all_vars if 'dis' in v.name]

        if self.model == 'DCGAN':
            self.opt_d = tf.train.AdamOptimizer(self.lr,beta1=self.mm).minimize(self.d_loss,var_list=d_vars)
            self.opt_g = tf.train.AdamOptimizer(self.lr, beta1=self.mm).minimize(self.g_loss, var_list=g_vars)
        elif self.model == 'WGAN_GP':
            self.opt_d = tf.train.AdamOptimizer(1e-5, beta1=0.5,beta2=0.9).minimize(self.d_loss, var_list=d_vars)
            self.opt_g = tf.train.AdamOptimizer(1e-5, beta1=0.5,beta2=0.9).minimize(self.g_loss, var_list=g_vars)
        else:
            print('model can only be "DCGAN","WGAN-GP" !')
            return

        self.saver = tf.train.Saver()
        load_model = False
        if not load_model:
            init = tf.global_variables_initializer()
            self.sess.run(init)
        elif load_model:
            self.saver.restore(self.sess, os.getcwd + '/model_saved/model.ckpt')


    def train(self):
        data_dir = os.getcwd() + '/64_crop'
        noise = np.random.uniform(-1,1,[self.batch_size,128])
        for epoch in range(self.EPOCH):
            iters = int(156191//self.batch_size)
            for idx in range(iters):
                start_t = time.time()
                real_img = load_data(data_dir,self.batch_size,idx=idx)
                train_img = np.array(real_img)
                g_opt = [self.opt_g,self.g_loss]
                d_opt = [self.opt_d,self.d_loss]
                feed = {self.x:train_img,self.z:noise}
                # update the Discrimater once
                _, loss_d = self.sess.run(d_opt, fee_dict=feed)
                # update the Generator twice
                _, loss_g = self.sess.run(g_opt, feed_dict=feed)
                _, loss_g = self.sess.run(g_opt, feed_dict=feed)

                print("[%3f][epoch:%2d/%2d][iter:%4d/%4d],loss_d:%5f,loss_g:%5f"%
                      (time.time()-start_t),epoch,self.EPOCH,idx,iters,loss_d,loss_g)
                plot.plot('d_loss', loss_d)
                plot.plot('g_loss', loss_g)

                if ((idx+1) % 100 ) == 0:#flush plot picture per 1000 iters
                    plot.flush()
                plot.tick()

                if (idx+1) % 400 == 0:
                    print('image saving ...........')
                    img = self.run(self.G_img,feed_dict=feed)
                    save_img.save_images(img,os.getcwd()+'/gen_picture/'+'samle{}_{}.jpg'.format(epoch,(idx+1)/400))
                    print('images save done')
            print('model saving ..........')
            path = os.getcwd() + '/model_saved'
            save_path = os.path.join(path,'model_ckpt')
            self.saver.save(self.sess,save_path=save_path)
            print('model saved .............')



