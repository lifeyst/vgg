
import matplotlib.pyplot as plt
import os
import os.path
import time
import numpy as np
import tensorflow as tf
import math
import input_data
import VGG
import tools
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import io
N_CLASSES = 9
BATCH_SIZE = 16
IS_PRETRAIN = True
def evaluate1():

    with tf.Graph().as_default():
        log_dir = './logs/train_st2/'  # 'C:/3_5th/VGG_model/logs/train/'  #训练日志，即训练参数
        # test_dir = './/cifar10_data//cifar-10-batches-bin//'
        test_dir = './data'
        n_test = 3000

        input_images, input_labels, input_labels1 = input_data.read_cifar10(data_dir=test_dir,
                                                                            is_train=True,
                                                                            batch_size=BATCH_SIZE,
                                                                            shuffle=True)
        # mean_data = np.mean(mnist.train.images, axis=0)
        logits, features = VGG.VGG16N(input_images, N_CLASSES, IS_PRETRAIN)
        correct = tools.num_correct_prediction(logits, input_labels)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("找到文件啦")
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print("没有找到文件")
                print('No checkpoint file found')
                return

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                print('\nEvaluating......')
                # j=0
                d = []
                tt = np.array([])

                for step in np.arange(1):
                    if coord.should_stop():
                        break
                    feat = np.array([])
                    images, labels = sess.run([input_images, input_labels])
                    mean_data = np.mean(images, axis=0)
                    # label=[]
                    # np.concatenate((array,q),axis=0)
                    b = np.transpose(np.nonzero(labels))[:, 1]
                    # d = b
                    d = np.concatenate((d, b), axis=0)
                    # labels=labels.tolist()
                    # print(d)
                    if step == 0:
                        tt = sess.run(features, feed_dict={input_images: images - mean_data})
                    else:
                        feat = sess.run(features, feed_dict={input_images: images - mean_data})

                    if step != 0:
                        tt = np.concatenate([tt, feat])



                fig = plt.figure(figsize=(16, 9))
                # fig = plt.figure()
                # ax = Axes3D(fig)
                #aa = TSNE(n_components=2).fit_transform(tt)
                pca = PCA(n_components=2)
                pca.fit(tt)
                aa = pca.transform(tt)
                #io.savemat('zongtnse.mat', {'matrix': aa})
                #lda = LinearDiscriminantAnalysis(n_components=2)
                #lda.fit(tt, d)
                #aa = lda.transform(tt)
                np.save('save_pca', aa)
                #aa = TSNE(n_components=2).fit_transform(tt)

                #print(aa[d==0,0].flatten())
                #np.save('vgg-9', aa)
                # ax.scatter(aa[:,0],aa[:,1],aa[:,2],c=labels1)
                # f = plt.figure()
                c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900']
                for i in range(9):
                    plt.plot((aa[d == i, 0].flatten())/100.0, (aa[d == i, 1].flatten())/100.0, '.', markersize=10, c=c[i])
 
                plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9'])
                # plt.xlim(-10, 10)
                # plt.ylim(-10,10)
                plt.grid()
                plt.show()
                # plt.close(fig)

            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)

if __name__ == '__main__':
    evaluate1()