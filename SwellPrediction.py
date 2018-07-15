import tensorflow as tf
from dbmodels import Wolpo

def RnnLinearCombined_model(features,labels,mode,params):
    seq=features['sequence']
    seq_onehot=tf.one_hot(seq)
    weather=features['weather']

    seq_len=params['seq_len']

    cell=tf.nn.rnn_cell.GRUCell(3)

    encoding=tf.nn.static_rnn(rnn_cell,seq_onehot)



    input=tf.concat(encoding,features['dense'])

    logits=tf.layers.dense(input,3)
    predicted=tf.argmax(logits,1)
    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'class':predicted
                }
            )
    loss=tf.losses.sparse_softmax_cross_entropy(labels,logits)
    if mode==tf.estimator.ModeKeys.TRAIN:
        optimizer=tf.train.AdadeltaOptimizer()
        train_op=optimizer.minimize(loss,tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode,loss=loss,train_op=train_op)

    eval_metric_ops={
        'accuracy':tf.metrics.accuracy(labels,predicted)
        }


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

