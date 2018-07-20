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


while current_date.year<2018:
    weather=guryong[guryong.moment==current_date].iloc[0].tolist()
    weather.pop(0)

    pred_date=current_date+timedelta(1)
    swells=swell_data[swell_data.moment==pred_date]
