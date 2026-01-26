from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()


def get_env(name):
    return os.getenv(name)


class Settings(BaseSettings):
    DATABASE_URL: str = get_env("DATABASE_URL")
    SOURCE_DIR: str = get_env("SOURCE_DIR")
    OPENAI_API_KEY: str = get_env("OPENAI_API_KEY")
    HF_TOKEN: str = get_env("HF_TOKEN")
    PINECONE_API_KEY: str = get_env("PINECONE_API_KEY")

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()

def get_settings() -> Settings:
    return settings