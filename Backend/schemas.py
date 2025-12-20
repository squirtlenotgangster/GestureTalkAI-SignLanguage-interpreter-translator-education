from pydantic import BaseModel

# --- User Schemas ---
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str  # We switched this from email to username to match your HTML
    password: str

class UserResponse(BaseModel):
    username: str
    email: str
    class Config:
        from_attributes = True

# --- Contact Form Schemas ---
class ContactCreate(BaseModel):
    name: str
    email: str
    message: str