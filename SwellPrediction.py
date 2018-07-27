import tensorflow as tf
from dbmodels import Wolpo
import pickle
import numpy as np
import json

train_path='swell_train.tfrecord'
test_path='swell_test.tfrecord'

def input_fn(files,num_epochs,shuffle,batch_size):
    dataset=tf.data.TFRecordDataset(files)

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


def RnnLinearCombined_model(features,labels,mode,params):
    seq=features['sequence']
    seq_onehot=tf.one_hot(seq,3,dtype=tf.float64)
    weather=features['weather']
    if mode==tf.estimator.ModeKeys.PREDICT:
        weather=tf.cast(weather,tf.float32)
    cells=[tf.nn.rnn_cell.GRUCell(3) for _ in range(4)]
    if 'use_dropout' in params.keys():
        if params['use_dropout'] and mode==tf.estimator.ModeKeys.TRAIN:
            cells=[tf.nn.rnn_cell.DropoutWrapper(elem,0.9,0.9) for elem in cells]
    layered_cell=tf.nn.rnn_cell.MultiRNNCell(cells)

    _,states=tf.nn.static_rnn(layered_cell,tf.unstack(tf.transpose(seq_onehot, perm=[1, 0, 2])),dtype=tf.float64)
    state=states[-1]
    state=tf.cast(state,tf.float32)
    input=tf.concat([state,weather],1)

    logits=tf.layers.dense(input,3)
    predicted=tf.argmax(logits,1,output_type=tf.int32)
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

def train_swell(use_dropout):
    tf.logging.set_verbosity(tf.logging.INFO)
    swell_classifier=tf.estimator.Estimator(RnnLinearCombined_model,'./models_final',tf.estimator.RunConfig(
        save_checkpoints_secs=15),params={'use_dropout':use_dropout})
    swell_classifier.train(lambda: input_fn([train_path,test_path],20,True,32),max_steps=20000)

def predict():
    with open('prediction.pkl','rb') as f:
        data=pickle.load(f)

    clf=tf.estimator.Estimator(RnnLinearCombined_model,'./models_final')
    result_list=[]
    for elem in data:
        date=elem[0]
        data=elem[1]
        weather=elem[2]
        temp_list=[]
        for i in range(24):
            temp_input_fn=tf.estimator.inputs.numpy_input_fn({'sequence':np.array([data[i:i+5]]),'weather':np.array([weather])},shuffle=False)
            result=next(clf.predict(temp_input_fn))['class']
            data.append(result)
            temp_list.append(int(result))
            temp_str=str(date.year)+'-'+str(date.month)+'-'+str(date.day)
        result_list.append({temp_str:temp_list})
    with open('answer.json','w') as f:
        json.dump(result_list,f)

def train_and_eval(use_dropout):
    tf.logging.set_verbosity(tf.logging.INFO)

    if use_dropout:
        model_path='./models_wdo'
    else:
        model_path='./models_wodo'

    swell_classifier=tf.estimator.Estimator(RnnLinearCombined_model,model_path,tf.estimator.RunConfig(
        save_checkpoints_secs=15),params={'use_dropout':use_dropout})

    train_spec=tf.estimator.TrainSpec(input_fn=lambda: input_fn([train_path],20,True,32))
    eval_Spec=tf.estimator.EvalSpec(input_fn=lambda: input_fn([test_path],1,False,4),throttle_secs=10)

    tf.estimator.train_and_evaluate(swell_classifier,train_spec,eval_Spec)

if __name__=='__main__':
    #train_and_eval(True)
    #train_swell(True)

    #predict()


    with open('prediction.pkl','rb') as f:
        data=pickle.load(f)

    for elem in data:
        print(elem[0])