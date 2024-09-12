from pydantic import BaseModel
from typing import List

class FaceBase(BaseModel):
    code: int
    name: str
    landmarks: List[float]  # Assuming landmarks is a list of float values

class FaceCreate(FaceBase):
    pass

class Face(FaceBase):
    id: int

    class Config:
        orm_mode = True