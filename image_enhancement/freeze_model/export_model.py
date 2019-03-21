import os
import tensorflow as tf
import tensorlayer as tl
from tensorflow.python.tools import freeze_graph
from model import SRGAN_g


# Uncomment the following line to print the GPU and tf and tl log
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.logging.set_verbosity(tf.logging.DEBUG)
# tl.logging.set_verbosity(tl.logging.DEBUG)

def preprocess(x):
    x = x / (255. / 2.)
    x = x - 1.
    return x


def export_model():
    """Load the model in TensorLayer's way and save
    the frozen graph

    Args:
        None

    Returns:
        None
    """

    # create folders to save result images
    checkpoint_dir = "checkpoint"

    ###========================== DEFINE MODEL ============================###
    t_image = tf.placeholder('float32', [None, None, None, 3],
                             name='input_image')
    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False))
    tl.layers.initialize_global_variables(sess)

    # Load model from .npz file
    tl.files.load_and_assign_npz(sess=sess,
                                 name=checkpoint_dir + '/g_srgan.npz',
                                 network=net_g)

    # export to meta file
    saver = tf.train.Saver()
    saver.save(sess, './meta/srgan')
    tf.train.write_graph(sess.graph.as_graph_def(), '.',
                         './meta/srgan.pbtxt', as_text=True)


if __name__ == "__main__":
    export_model()
    freeze_graph.freeze_graph('./meta/srgan.pbtxt', "", False,
                              './meta/srgan', "SRGAN_g/out/Tanh",
                              "save/restore_all", "save/Const:0",
                              '../frozen_model.pb', True, ""
                              )
