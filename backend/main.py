from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import select
from database import models, schemas, database

from typing import Optional, List

from ocrmac import ocrmac
import face_recognition
import os

app = FastAPI()

# Create database tables
models.Base.metadata.create_all(bind=database.engine)

# Dependency to provide a database session
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

def save_face_to_db(db: Session, name: str, code: int, encoding: list, image_path: str):
    # Проверяем, что лицо обнаружено
    if encoding is None:
        print("No face detected in the image.")
        return False, "No face detected in the image."

    # Проверяем, существует ли уже запись с таким же кодом
    existing_face = db.query(models.Face).filter(models.Face.code == code).first()
    if existing_face:
        print("Face with this code already exists in the database.")
        return False, "Face with this code already exists in the database."

    # Создаем новый объект Face
    new_face = models.Face(code=code, name=name, landmarks=encoding, picture=image_path)

    # Добавляем новый объект в базу данных
    db.add(new_face)
    db.commit()
    db.refresh(new_face)

    return new_face

def extract_face_encoding(image_path):
    # Load image
    image = face_recognition.load_image_file(image_path)
    
    # Find face locations
    face_locations = face_recognition.face_locations(image)
    
    # Get encodings for the faces in the image
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    if not face_encodings:
        return None
    
    # Assuming we are interested in the first face found
    return face_encodings[0]

def calculate_similarity(encoding1, encoding2):
    # Calculate similarity between two face encodings
    return face_recognition.face_distance([encoding1], encoding2)[0]

# Function to extract a code (like ID) from the image using OCR
def get_code(image_path):
    annotations = ocrmac.OCR(image_path).recognize()
    if not annotations:
        return None
    try:
        # Предполагаем, что annotations[0] содержит код как строку
        # Проверьте тип данных в annotations и извлеките строку из кортежа
        if isinstance(annotations[0], tuple):
            # Если это кортеж, возможно, код находится в первом элементе кортежа
            code_str = annotations[0][0]
        else:
            # Если это строка, используем её напрямую
            code_str = annotations[0]
        
        return int(code_str)  # Преобразуем строку в целое число
    except (ValueError, TypeError):
        return None



@app.post("/api/find_similar")
async def find_similar(image: UploadFile = File(...), db: Session = Depends(get_db)):
    # Сохранение загруженного изображения
    media_path = "media"
    if not os.path.exists(media_path):
        os.makedirs(media_path)
    
    image_path = os.path.join(media_path, image.filename)
    with open(image_path, "wb") as buffer:
        buffer.write(await image.read())

    # Извлечение лендмарков из загруженного изображения
    uploaded_face_encoding = extract_face_encoding(image_path)
    if uploaded_face_encoding is None:
        raise HTTPException(status_code=400, detail="No face detected in the image")

    # Поиск похожих лиц в базе данных
    faces = db.query(models.Face).all()
    similar_faces = []

    for face in faces:
        # Извлечение лендмарков для лиц из базы данных
        db_face_encoding = face_recognition.face_encodings(face_recognition.load_image_file(face.picture))[0]
        similarity = calculate_similarity(uploaded_face_encoding, db_face_encoding)
        
        if similarity < 0.6:  # Пример порога
            similar_faces.append({
                "id": face.id,
                "name": face.name,
                "code": face.code,
                "landmarks": face.landmarks,  # Обратите внимание на обработку бинарных данных
                "picture": face.picture
            })

    if not similar_faces:
        return {"message": "No similar faces found"}

    return similar_faces  # Ensure that `as_dict` method exists or adapt as needed


# API endpoint to upload an image, extract face data, and save to the database
@app.post("/api/upload_user")
async def upload_image(image: UploadFile = File(...), db: Session = Depends(get_db)):
    # Save the uploaded image to a media folder
    media_path = "media"
    if not os.path.exists(media_path):
        os.makedirs(media_path)
    
    image_path = os.path.join(media_path, image.filename)
    with open(image_path, "wb") as buffer:
        buffer.write(await image.read())

    # Extract the code from the image
    code = get_code(image_path)
    if code is None:
        raise HTTPException(status_code=400, detail="Unable to extract code from the image or code is not a valid integer")

    # Check if the code is unique
    existing_face = db.query(models.Face).filter(models.Face.code == code).first()
    if existing_face:
        raise HTTPException(status_code=400, detail="Code already exists in the database")

    # Extract face encoding from the image
    face_encoding = extract_face_encoding(image_path)
    if face_encoding is None:
        raise HTTPException(status_code=400, detail="No face detected in the image")

    # Save the extracted data to the database
    face = save_face_to_db(db=db, name=image.filename, code=code, encoding=face_encoding.tolist(), image_path=image_path)

    # Return the saved face data
    return face

@app.get("/api/faces/code/")
async def get_faces_by_code(code: Optional[int] = None, db: Session = Depends(get_db)):
    if code is not None:
        face = db.query(models.Face).filter(models.Face.code == code).first()
        if face is None:
            raise HTTPException(status_code=404, detail="Face not found")
        return {
            "id": face.id,
            "name": face.name,
            "code": face.code,
            "landmarks": face.landmarks,  # Ensure proper handling of binary data
            "picture": face.picture
        }
    else:
        faces = db.query(models.Face).all()
        return [
            {
                "id": face.id,
                "name": face.name,
                "code": face.code,
                "landmarks": face.landmarks,  # Ensure proper handling of binary data
                "picture": face.picture
            }
            for face in faces
        ]