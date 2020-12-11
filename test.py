import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from modelDSG import CycleGAN
from datetime import datetime
import os
import logging
#import string
import random
import numpy as np
import aug
from utils import ImagePool
import SimpleITK as sitk
import scipy.io as sio 
import time
try:
  from os import scandir
except ImportError:
  # Python 2 polyfill module
  from scandir import scandir
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
FLAGS = tf.flags.FLAGS
from metric import dice_score2 
from metric import sensitivity 
from metric import haff 
from metric import dice_score2 
tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 400, 'image size, default: 400')
tf.flags.DEFINE_bool('use_lsgan', True,
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'batch',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_integer('lambda1', 2,
                        'weight for forward cycle loss (X->Y->X), default: 10')
tf.flags.DEFINE_integer('lambda2', 2,
                        'weight for backward cycle loss (Y->X->Y), default: 10')
tf.flags.DEFINE_float('learning_rate', 1e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50,
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_integer('class_num', 8,
                        'number of gen filters in first conv layer, default: 64')
                        
tf.flags.DEFINE_string('DataPath', '/home/libo/GANS/DATA/DH2NEO_TEST/',
                       'X tfrecords file for training, default:')
#tf.flags.DEFINE_string('Y', 'facade/tfrecords/Y.tfrecords',
                       #'Y tfrecords file for training, default:')
tf.flags.DEFINE_string('Z', 'tfrecords/Z.tfrecords',
                       'Z tfrecords file for training, default:')
tf.flags.DEFINE_string('load_model','20201129-0853',
                        'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
                        
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
OutputPath = '/home/libo/GANS/DH2NEO_AXIAL/result_label/'

def data_reader(input_dir, shuffle=True):
  """Read images from input_dir then shuffle them
  Args:
    input_dir: string, path of input dir, e.g., /path/to/dir
  Returns:
    file_paths: list of strings
  """
  file_paths = []

  for img_file in scandir(input_dir):
    if img_file.name.endswith('.jpg') and img_file.is_file():
      file_paths.append(img_file.path)
    if img_file.name.endswith('.png') and img_file.is_file():
      file_paths.append(img_file.path)
    if img_file.name.endswith('.mat') and img_file.is_file():
      file_paths.append(img_file.path)

  if shuffle:
    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(file_paths)))
#    random.seed(12345)
    random.shuffle(shuffled_index)

    file_paths = [file_paths[i] for i in shuffled_index]

  return file_paths
def train():
  if FLAGS.load_model is not None:
    checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
#    checkpoints_dir2 = "checkpoints2/" + FLAGS.load_model.lstrip("checkpoints2/")
  else:
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    checkpoints_dir = "checkpoints/{}".format(current_time)
#    checkpoints_dir2 = "checkpoints2/{}".format(current_time)
    try:
      os.makedirs(checkpoints_dir)
    except os.error:
      pass
  
  graph = tf.Graph()
  with graph.as_default():
    cycle_gan = CycleGAN(
        is_trainGAN = True,
        is_trainUnet = True,
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.image_size,
        use_lsgan=FLAGS.use_lsgan,
        norm=FLAGS.norm,
        lambda1=FLAGS.lambda1,
        lambda2=FLAGS.lambda2,
        learning_rate=FLAGS.learning_rate,
        beta1=FLAGS.beta1,
        ngf=FLAGS.ngf,
        num_class = FLAGS.class_num
    )
    G_gan_loss_z, D_Z_loss,fake_y,DiceLoss,DiceTrain,seg,diceSIM = cycle_gan.model()
#    optimizersU = cycle_gan.optimize2(DiceTrain)
#    optimizers = cycle_gan.optimize(G_gan_loss_z,D_Z_loss,DiceLoss)
#    optimizers3 = cycle_gan.optimize3(G_gan_loss_z,DiceLoss)
    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
    saver = tf.train.Saver()
  tf_config = tf.ConfigProto()  
#  tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
  #with tf.Session(graph=graph,config=tf.ConfigProto(log_device_placement=True)) as sess:
  with tf.Session(graph=graph) as sess:
    if FLAGS.load_model is not None:
      checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
      meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
      restore = tf.train.import_meta_graph(meta_graph_path)
      restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
      print(meta_graph_path)
      step = int(meta_graph_path.split("-")[2].split(".")[0])
    else:
      sess.run(tf.global_variables_initializer())
      step = 0

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    accsum = np.zeros((1,8),dtype = np.float32)
    accsensi = np.zeros((1,8),dtype = np.float32)
    acchaff = np.zeros((1,8),dtype = np.float32)
    try:
      fake_Y_pool = ImagePool(FLAGS.pool_size)
#      fake_X_pool = ImagePool(FLAGS.pool_size)
#      fake_Z_pool = ImagePool(FLAGS.pool_size)
      loss_1000 = 0
      loss_echo = 0
      while not coord.should_stop():
#        a = time.time()
        file_paths_X = data_reader(FLAGS.DataPath)
        count = 0
        label_gt = np.zeros((50,400,400),dtype = np.float32)
        label_pred = np.zeros((50,400,400),dtype = np.float32)
        imgy = np.zeros((50,400,400),dtype = np.float32)
        k = 0
        
        for i in range(len(file_paths_X)):
          file_paths_X.sort()
          file_path_X = file_paths_X[i]
#          print (file_path_X)
          data=sio.loadmat(file_path_X) 
          y_val = np.float32(data['y'])
          y_val = y_val.astype(np.float32)
          imgy[k,:,:] = y_val
          
          z_val = np.float32(data['z'])
          z_val = np.squeeze(z_val)
    
          x_val = np.float32(data['x'])
          x_val = np.squeeze(x_val)
          label_gt[k,:,:] = x_val
#            x_val,y_val,z_val = aug.aug2D(x_val,y_val,z_val,FLAGS.class_num)
          x_val = x_val[np.newaxis,:,:];y_val = y_val[np.newaxis,:,:,np.newaxis];z_val = z_val[np.newaxis,:,:,np.newaxis]
#            sio.savemat(OutputPath+str(count)+'.mat',{'x':x_val,'y':y_val},do_compression = True)
#            count +=1
#            continue
          seg_val = (sess.run(seg, feed_dict={cycle_gan.y: y_val}))
          seg_val = np.squeeze(seg_val)
          inds = np.argmax(seg_val,axis = 2)
          inds = inds.astype(np.int32)
          label_pred[k,:,:] = inds
          k += 1
          
          if k == 50:
            acc = dice_score2(label_gt, label_pred, 8)
            accsum = accsum + acc.copy()
            acc = sensitivity(label_gt, label_pred, 8)
            accsensi = accsensi + acc.copy()
            acc = haff(label_gt, label_pred, 8)
            acchaff = acchaff + acc.copy()
            if file_path_X.rsplit('/')[-1].split('_')[0] == '1f':
              resultImage = sitk.GetImageFromArray(label_pred)
              sitk.WriteImage(resultImage, OutputPath + file_path_X.rsplit('/')[-1].split('_')[0] + '.nii')
#              resultImage = sitk.GetImageFromArray(label_gt)
#              sitk.WriteImage(resultImage, OutputPath + file_path_X.rsplit('/')[-1].split('_')[0] + 'gt.nii')
#              resultImage = sitk.GetImageFromArray(imgy)
#              sitk.WriteImage(resultImage, OutputPath + file_path_X.rsplit('/')[-1].split('_')[0] + 'y.nii')
              break
            label_gt = np.zeros((50,400,400),dtype = np.float32)
            label_pred = np.zeros((50,400,400),dtype = np.float32)
            
            k = 0
        accsum = accsum / 4
        accsensi = accsensi / 4
        acchaff = acchaff / 4
        print(accsum)
        print(accsensi)
        print(acchaff)
        break
    except KeyboardInterrupt:
      logging.info('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
#      save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
#      logging.info("Model saved in file: %s" % save_path)
      # When done, ask the threads to stop.
      coord.request_stop()
      coord.join(threads)

def main(unused_argv):
  train()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()
