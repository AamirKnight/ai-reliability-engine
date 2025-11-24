from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Gemini 3 Reliability Engine"
    
    # We use the standard variable name for the new SDK
    GEMINI_API_KEY: str 
    
    # UPGRADE: Using the latest 2025 Reasoning Model
    GEMINI_MODEL: str = "gemini-3-pro-preview" 

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()