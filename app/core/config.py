from pydantic_settings import BaseSettings, SettingsConfigDict

# app/core/config.py

class Settings(BaseSettings):
    PROJECT_NAME: str = "Gemini Reliability Engine"
    GEMINI_API_KEY: str 
    
    # 🔴 CHANGE THIS:
    # GEMINI_MODEL: str = "gemini-3-pro-preview" 
    
    # 🟢 TO THIS (The fastest, most reliable current model):
    GEMINI_MODEL: str = "gemini-2.5-flash"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()