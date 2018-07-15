from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column,Float,Integer,Date

Base=declarative_base()

class Wolpo(Base):
    __tablename__='wolpo'

    id=Column(Integer,primary_key=True)
    moment=Column(Date)
    temp_avg2=Column(Float)
    temp_max2=Column(Float)
    temp_low2=Column(Float)
    height_w_avg2=Column(Float)
    weight_avg2=Column(Float)
    height_w_max2=Column(Float)
    height_max2=Column(Float)
    frequency_avg2=Column(Float)
    frequency_max2=Column(Float)