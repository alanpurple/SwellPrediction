from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column,Float,Integer,Date

Base=declarative_base()

class Guryong(Base):
    __tablename__='guryong'

    id=Column(Integer,primary_key=True)
    moment=Column(Date)
    temp_avg1=Column(Float)
    temp_max1=Column(Float)
    temp_low1=Column(Float)
    height_w_avg1=Column(Float)
    weight_avg1=Column(Float)
    height_w_max1=Column(Float)
    height_max1=Column(Float)
    frequency_avg1=Column(Float)
    frequency_max1=Column(Float)