from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column,Float,Integer,Date

Base=declarative_base()

class Swell(Base):
    __tablename__='swell'

    id=Column(Integer,primary_key=True)
    moment=Column(Date)
    from_time=Column(Integer)
    to_time=Column(Integer)
    type=Column(Integer)