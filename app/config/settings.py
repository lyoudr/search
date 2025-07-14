from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()


def get_env(name):
    return os.getenv(name)


class Settings(BaseSettings):
    DATABASE_URL: str = get_env("DATABASE_URL")
    SOURCE_DIR: str = get_env("SOURCE_DIR")
    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()

def get_settings() -> Settings:
    return settings