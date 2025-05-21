from os import getenv
from openai import AsyncOpenAI

openai_client = AsyncOpenAI(
    api_key=getenv("OPENAI_API_KEY")
).with_options(
    timeout=20,
    max_retries=0
)