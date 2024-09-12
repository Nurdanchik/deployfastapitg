from sqlalchemy import Column, Integer, String, JSON
from .database import Base

class Face(Base):
    __tablename__ = "faces"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(Integer, unique=True, nullable=False)
    name = Column(String, index=True)
    landmarks = Column(JSON, unique=False)  # Storing landmarks as JSON
    picture = Column(String)

    def as_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "code": self.code,
            "landmarks": self.landmarks,  # You may need to handle the encoding if you return this field
            "picture": self.picture
        }