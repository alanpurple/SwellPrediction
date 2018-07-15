import tensorflow as tf
from datetime import date,timedelta
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

temp=guryong[guryong.moment=='2014-11-01'].iloc[0].tolist()
temp.pop(0)

current_date=date(2014,1,1)