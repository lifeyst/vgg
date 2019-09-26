#%%
#DATA:
    #1. cifar10(binary version):https://www.cs.toronto.edu/~kriz/cifar.html
    #2. pratrained weights (vgg16.npy):https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM
    
# TO Train and test:
    #0. get data ready, get paths ready !!!
    #1. run training_and_val.py and call train() in the console
    #2. call evaluate() in the console to test
    
#%%
#from scipy import io
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
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#export CUDA_VISIBLE_DEVICES=1
filename = 'loss_acc.txt'
#%%
IMG_W = 224
IMG_H = 224
N_CLASSES = 9
BATCH_SIZE = 16
learning_rate = 0.001
#LAMBDA1= 0.001
LAMBDA2 = 0.005
CENTER_LOSS_ALPHA=0.5

MAX_STEP = 30050   #15000 it took me about one hour to complete the training.
IS_PRETRAIN = True

#export CUDA_VISIBLE_DEVICE= 0
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
#%%   Training

def train():

#    pre_trained_weights = './/vgg16_pretrain//vgg16.npy'  #存放预训练的权值
    pre_trained_weights = './/retrain//vgg16.npy'
#    data_dir = './/data//cifar-10-batches-bin//'  #存放数据，相对路径的用法
    data_dir = './/data//'  #存放数据,绝对路径的用法
    train_log_dir = './/logs//train_st5//'  #存放训练日志,相对路径
    val_log_dir = './/logs//val_st5//'    #存放测试日志,相对路径
    #my_='C:\\tem'  #这个路径是用来存放tensorboard数据的
    with tf.name_scope('input'):
        tra_image_batch, tra_label_batch, tra_label_batch1 = input_data.read_cifar10(data_dir=data_dir,
                                                 is_train=True,
                                                 batch_size= BATCH_SIZE,
                                                 shuffle=True)
#        b=np.array(tra_image_batch)
#        c=np.array(tra_label_batch)
#        print(b,c)
        val_image_batch, val_label_batch, val_label_batch1 = input_data.read_cifar10(data_dir=data_dir,
                                                 is_train=False,
                                                 batch_size= BATCH_SIZE,
                                                 shuffle=False)

#    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
#    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES])
#    b=np.array(x)
#    c=np.array(y_)
#    print(b,c)
    #logits,features = VGG.VGG16N(tra_image_batch, N_CLASSES, IS_PRETRAIN)
    logits,features = VGG.VGG16N(tra_image_batch, N_CLASSES, IS_PRETRAIN)
    #loss,centers_update_op = tools.total_loss(logits, tra_label_batch,features, BATCH_SIZE,N_CLASSES,CENTER_LOSS_ALPHA,ratio1=LAMBDA1,ratio2=LAMBDA2)
    #loss = tools.total_loss(logits, tra_label_batch,features, BATCH_SIZE,ratio=LAMBDA1)
    loss,centers_update_op = tools.total_loss(logits,tra_label_batch,features, CENTER_LOSS_ALPHA, N_CLASSES,ratio=LAMBDA2)



    #loss = tools.loss(logits, tra_label_batch)
    accuracy = tools.accuracy(logits, tra_label_batch)

    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.control_dependencies([centers_update_op]):
        train_op = tools.optimize(loss, learning_rate, my_global_step)

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    #config = tf.ConfigProto(allow_soft_placement=True)
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    #config.gpu_options.allow_growth = True
    #sess = tf.Session(config=config)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8 # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True #allocate dynamically
    sess = tf.Session(config = config)

    #sess = tf.Session()
    sess.run(init)

    # load the parameter file, assign the parameters, skip the specific layers
    tools.load_with_skip(pre_trained_weights, sess, ['fc6','fc7','fc8'])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break

            tra_images,tra_labels = sess.run([tra_image_batch, tra_label_batch])
            mean_data = np.mean(tra_images, axis=0)
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={tra_image_batch:tra_images-mean_data ,tra_label_batch:tra_labels})
            if step % 10 == 0 or (step ) == MAX_STEP:
                print ('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                #with open(filename, 'a') as file_object:
                    #file_object.write(str(step) + " "+str(tra_loss)+" "+str(tra_acc)+"\n")
                summary_str = sess.run(summary_op)
                tra_summary_writer.add_summary(summary_str, step)
                

            if step % 200 == 0 or (step ) == MAX_STEP:
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={tra_image_batch:val_images-mean_data,tra_label_batch:val_labels})
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc))


                summary_str = sess.run(summary_op)
                val_summary_writer.add_summary(summary_str, step)
                    
            if step % 2000 == 0 or (step ) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()

#%%   Test the accuracy on test dataset. got about 85.69% accuracy.


def evaluate():
    with tf.Graph().as_default():
        log_dir='./logs/train_st6/'   #'C:/3_5th/VGG_model/logs/train/'  #训练日志，即训练参数
        #test_dir = './/cifar10_data//cifar-10-batches-bin//'
        test_dir='./data4'
        n_test = 1000

        images, labels = input_data.read_cifar10(data_dir=test_dir,
                                                    is_train=False,
                                                    batch_size= BATCH_SIZE,
                                                    shuffle=False)

        logits,features = VGG.VGG16N(images, N_CLASSES, IS_PRETRAIN)
        correct = tools.num_correct_prediction(logits, labels)
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
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            
            try:
                print('\nEvaluating......')
                start = time.time()
                num_step = int(math.floor(n_test / BATCH_SIZE))
                num_sample = num_step*BATCH_SIZE
                step = 0
                total_correct = 0
                while step < num_step and not coord.should_stop():
                    batch_correct = sess.run(correct)
                    total_correct += np.sum(batch_correct)
                    step += 1
                print('Total testing samples: %d' %num_sample)
                print('Total correct predictions: %d' %total_correct)
                print('Average accuracy: %.2f%%' %(100*total_correct/num_sample))
                end = time.time()
                TrnTime = end - start
                print('PCANet training time:%f' % TrnTime)
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)


def evaluate1():
    '''
    def pca(XMat, k):
        average = np.mean(XMat,axis=0) 
        m, n = np.shape(XMat)
        data_adjust = []
        avgs = np.tile(average, (m, 1))
        data_adjust = XMat - avgs
        covX = np.cov(data_adjust.T)   #计算协方差矩阵
        featValue, featVec=  np.linalg.eig(covX)  #求解协方差矩阵的特征值和特征向量
        index = np.argsort(-featValue) #依照featValue进行从大到小排序
        finalData = []
        if k > n:
           print ("k must lower than feature number")
           return
        else:
        #注意特征向量时列向量。而numpy的二维矩阵(数组)a[m][n]中，a[1]表示第1行值
            selectVec = np.matrix(featVec.T[index[:k]]) #所以这里须要进行转置
            finalData = data_adjust * selectVec.T 
            reconData = (finalData * selectVec) + average  
        return finalData, reconData
    '''
    with tf.Graph().as_default():
        log_dir = './logs/train_st3/'  # 'C:/3_5th/VGG_model/logs/train/'  #训练日志，即训练参数
        # test_dir = './/cifar10_data//cifar-10-batches-bin//'
        test_dir = './data'
        n_test = 3000

        input_images, input_labels,input_labels1 = input_data.read_cifar10(data_dir=test_dir,
                                                 is_train=True,
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=True)
        #mean_data = np.mean(mnist.train.images, axis=0)
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
                #j=0
                d = []
                tt=np.array([])
                
                for step in np.arange(200):
                    if coord.should_stop():
                        break
                    feat=np.array([])
                    images, labels = sess.run([input_images,input_labels])
                    mean_data = np.mean(images, axis=0)
                    #label=[]
                    #np.concatenate((array,q),axis=0)
                    b = np.transpose(np.nonzero(labels))[:,1]
                    #d = b
                    d=np.concatenate((d,b),axis=0)
                #labels=labels.tolist()
                    #print(d)
                    if step==0:
                        tt = sess.run(features, feed_dict={input_images: images-mean_data})
                    else:
                        feat = sess.run(features, feed_dict={input_images: images-mean_data})
                    
                    if step!=0:
                        tt = np.concatenate([tt, feat])
                
                    #aa,bb=pca(tt, 2)
    
                    #aa=pca_via_svd(tt, 2)
                    #tt=np.row_stack((tt,feat))
                    #j=1
                    #tt=np.concatenate([tt,feat])
                    #io.savemat('qingqianfc6.mat', {'matrix': tt})
                    #print(feat)
                #print('images:',images)
                #print(d)   
                #label = labe
                #labels1 = tf.argmax(labels, 1)
                fig = plt.figure(figsize=(16, 9))
                #fig = plt.figure()
                #ax = Axes3D(fig)
                pca = PCA(n_components=2)
                pca.fit(tt)
                aa=pca.transform(tt)
                #ax.scatter(aa[:,0],aa[:,1],aa[:,2],c=labels1)
                #f = plt.figure()
                c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff']
                for i in range(5):
                    plt.plot(aa[d == i, 0].flatten(), aa[d == i, 1].flatten(), '.',markersize=10, c=c[i])
                    #print(tt[d==i,0].flatten())
                    #print(feat[d==i,1].flatten())
                    #print(d)
                plt.legend(['Aircaftcarrier', 'cargoship', 'fightship', 'cruiseship', 'submarine'])
                #plt.xlim(-10, 10)
                #plt.ylim(-10,10)
                plt.grid()
                plt.show()
                #plt.close(fig)

            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)


def evaluate2():
    with tf.Graph().as_default():
        log_dir = './logs/train_st5/'  # 'C:/3_5th/VGG_model/logs/train/'  #训练日志，即训练参数
        # test_dir = './/cifar10_data//cifar-10-batches-bin//'
        test_dir = './data4'
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
                #outputs = []
                labels = []
                feats = []
                for i in range(testNums // 128 + 1):
                    # here sess.run a batch data
                    batch_features, images,test_labs = sess.run([test_features, input_images, input_labels])
                    #outputs.extend(batch_logits.tolist())
                    labels.extend(test_labs.tolist())
                    feats.extend(batch_features.tolist())
                features = np.array(features)
                fig = plt.figure()
                ax = Axes3D(fig)
                pca = PCA(n_components=2)
                features = pca.fit_transform(features)
                ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels)

                # j=0
                d = []
                tt = np.array([])

                for step in np.arange(200):
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
                        tt = sess.run(features, feed_dict={input_images: images })
                    else:
                        feat = sess.run(features, feed_dict={input_images: images })

                    if step != 0:
                        tt = np.concatenate([tt, feat])
                    # pca = PCA(n_components=2)
                    # pca.fit(tt)
                    # aa=pca.transform(tt)
                    #aa, bb = pca(tt, 2)

                    # aa=pca_via_svd(tt, 2)
                    # tt=np.row_stack((tt,feat))
                    # j=1
                    # tt=np.concatenate([tt,feat])
                    # io.savemat('qingqianfc6.mat', {'matrix': tt})
                    # print(feat)
                # print('images:',images)
                # print(d)
                # label = labe

                f = plt.figure(figsize=(16, 9))
                # f = plt.figure()
                c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff']
                for i in range(5):
                    plt.plot(aa[d == i, 0].flatten(), aa[d == i, 1].flatten(), '.', markersize=10, c=c[i])
                    # print(tt[d==i,0].flatten())
                    # print(feat[d==i,1].flatten())
                    # print(d)
                plt.legend(['Aircaftcarrier', 'cargoship', 'fightship', 'cruiseship', 'submarine'])
                # plt.xlim(-10, 10)
                # plt.ylim(-10,10)
                plt.grid()
                plt.show()

            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)


if __name__ == '__main__':
    train()
    #start = time.time()
    #evaluate()
    #evaluate1()
    #end = time.time()
    #TrnTime = end - start
    #print('PCANet training time:%f' % TrnTime)
