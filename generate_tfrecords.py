import tensorflow as tf
from datetime import timedelta,datetime
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base
# from dbmodels import Wolpo,Swell

seq_len=5
is_pre=True

server = 'localhost'
database = 'steroid'
driver = 'MySQL ODBC 5.3 Unicode Driver'
id = 'root'
pwd = 'alan'
# Base=automap_base()
engine=create_engine("mysql+mysqlconnector://{}:{}@{}/{}".format(id,pwd,server,database))
# Session=sessionmaker(engine)
# session=Session()
# Base.prepare(engine,reflect=True)
swell_data=pd.read_sql_table('swell',engine,index_col='id')

# swell_data=session.query(Base.classes.swell)

guryong=pd.read_sql_table('guryong',engine,index_col='id')

current_date=datetime(2014,1,1)

#temp_date=datetime(2014,11,1)
#print(guryong.dtypes)
#temp=guryong[guryong.moment==temp_date].iloc[0].tolist()
#print(temp)
#print(temp.pop(0))
#print(temp)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_list_feature(elems):
    return tf.train.Feature(float_list=tf.train.FloatList(value=elems))

goal_dates=[]

train_writer=tf_writer=tf.python_io.TFRecordWriter('swell_train.tfrecord')
test_writer=tf_writer=tf.python_io.TFRecordWriter('swell_test.tfrecord')

EVAL_DATES=[datetime(2014,3,14),datetime(2014,6,12),datetime(2014,8,13),datetime(2014,11,25),\
    datetime(2015,2,28),datetime(2015,8,25),datetime(2015,9,11),datetime(2015,11,28),datetime(2016,2,15),\
    datetime(2016,4,21),datetime(2016,9,23),datetime(2016,12,21),datetime(2017,1,15),datetime(2017,5,24),datetime(2017,8,14),\
    datetime(2017,11,10)]

total_data=[]

while current_date.year<2018:
    weather=guryong[guryong.moment==current_date].iloc[0].tolist()
    weather.pop(0)

    pred_date=current_date+timedelta(1)
    swells=swell_data[swell_data.moment==pred_date]
    swells_prev=swell_data[swell_data.moment==current_date]
    prev_data=[]
    # Use the fact that data is sorted in time-ascending
    if len(swells_prev)==0:
        prev_data=[0]*5
    elif len(swells_prev)==1:
        temp=swells_prev.iloc[0]
        if temp['from_time']==0:
            prev_data=[-1]*5
        elif temp['from_time']<26:
            if temp['to_time']==31:
                prev_data=[temp['type']]*5
            elif temp['to_time']>26:
                prev_data=[temp['type']]*(temp['to_time']-26)+[0]*(31-temp['to_time'])
            else:
                prev_data=[0]*5
        else:
            prev_data=[0]*5
    else:
        prev_offset=0
        time_prev=swells_prev.iloc[0]['from_time']
        prev_data=[]
        for i in range(len(swells_prev)-1):
            if swells_prev.iloc[i]['to_time']>26:
                if swells_prev.iloc[i]['from_time']<26:
                    prev_data+=[swells_prev.iloc[i]['type']]*(swells_prev.iloc[i]['to_time']-26)
                else:
                    if swells_prev.iloc[i]['from_time']>26 and len(prev_data)==0:
                        prev_data+=[0]*(swells_prev.iloc[i]['from_time']-26)
                    prev_data+=[swells_prev.iloc[i]['type']]*(swells_prev.iloc[i]['to_time']-swells_prev.iloc[i]['from_time'])
        if len(prev_data)<5:
            prev_data+=[0]*(5-len(prev_data))

    # check for goal date
    if len(swells)==1 and swells.iloc[0]['from_time']==0:
        assert swells.iloc[0]['to_time']==0
        assert swells.iloc[0]['type']==4
        current_date+=timedelta(1)
        goal_dates.append(pred_date)
        continue
    elif len(swells)==0:
        pred_series=[0]*24
    else:
        pred_series=[]
        swells.sort_values(by=['from_time'])
        swell_data_offeset=0
        i=7
        while i<31:
            swell_fragment=swells.iloc[swell_data_offeset]
            if swell_fragment['from_time']>i:
                pred_series+=[0]*(swell_fragment['from_time']-i)
            pred_series+=[swell_fragment['type']]*(swell_fragment['to_time']-swell_fragment['from_time'])
            i=swell_fragment['to_time']
            if swell_data_offeset==len(swells)-1:
                if i<31:
                    pred_series+=[0]*(31-i)
            else:
                swell_data_offeset+=1

    current_date+=timedelta(1)

    total_data.append([weather,prev_data,swell_data])

print(len(total_data))


print(total_data[3])
print(total_data[10])