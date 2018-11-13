from train import Train
import tensorflow as tf
def main():
    sess = tf.Session()
    train = Train(sess)
    train.build_model()
    train.train()


if __name__ == '__main__':
  main()