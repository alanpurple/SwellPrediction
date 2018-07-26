import tensorflow as tf
from dbmodels import Wolpo

train_path='swell_train.tfrecord'
test_path='swell_test.tfrecord'

def input_fn(file,num_epochs,shuffle,batch_size):
    dataset=tf.data.TFRecordDataset([file])

    def parser(record):
        keys_to_features={
            'sequence':tf.FixedLenFeature((5),tf.int64),
            'weather':tf.FixedLenFeature((9),tf.float32),
            'label':tf.FixedLenFeature((1),tf.int64)
            }
        parsed=tf.parse_single_example(record,keys_to_features)
        label=parsed.pop('label')
        return parsed,label

    dataset=dataset.map(parser,4)
    if shuffle:
        dataset=dataset.shuffle(10000)
    dataset=dataset.batch(batch_size)
    dataset=dataset.repeat(num_epochs)

    iterator=dataset.make_one_shot_iterator()
    return iterator.get_next()


def RnnLinearCombined_model(features,labels,mode):
    seq=features['sequence']
    seq_onehot=tf.one_hot(seq,3,dtype=tf.float64)
    weather=features['weather']
    cells=[tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(3),0.9,0.9) for _ in range(4)]
    layered_cell=tf.nn.rnn_cell.MultiRNNCell(cells)

    _,states=tf.nn.static_rnn(layered_cell,tf.unstack(tf.transpose(seq_onehot, perm=[1, 0, 2])),dtype=tf.float64)
    state=states[-1]
    state=tf.cast(state,tf.float32)
    input=tf.concat([state,weather],1)

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
        optimizer=tf.train.AdamOptimizer(0.0005)
        train_op=optimizer.minimize(loss,tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode,loss=loss,train_op=train_op)

    eval_metric_ops={
        'accuracy':tf.metrics.accuracy(labels,predicted)
        }
    return tf.estimator.EstimatorSpec(mode,loss=loss,eval_metric_ops=eval_metric_ops)


if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    swell_classifier=tf.estimator.Estimator(RnnLinearCombined_model,'./models',tf.estimator.RunConfig())

    train_spec=tf.estimator.TrainSpec(input_fn=lambda: input_fn(train_path,15,True,32))
    eval_Spec=tf.estimator.EvalSpec(input_fn=lambda: input_fn(test_path,1,True,4))

    tf.estimator.train_and_evaluate(swell_classifier,train_spec,eval_Spec)


