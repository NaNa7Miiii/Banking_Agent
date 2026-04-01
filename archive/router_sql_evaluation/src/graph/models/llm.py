import os
import time
from pathlib import Path
from typing import Literal, Optional, Dict, Union
from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError, APIError, APIConnectionError, APITimeoutError

# load .env file from project root
project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(dotenv_path=project_root / '.env')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment or .env file")

DEFAULT_MODEL_CONFIG = {
    "frequency_penalty": 0,
    "max_tokens": 2048,
    "presence_penalty": 0,
    "top_p": 1,
}

client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=100,
    max_retries=3,
)

# Role-based LLM configuration
LLMRole = Literal["sql", "fraud", "rag", "tavily", "router", "summary", "generic"]
ROLE_DEFAULTS: Dict[LLMRole, Dict[str, Union[str, float, int]]] = {
    "sql":    {"model_name": "gpt-4.1-mini", "temperature": 0.0},
    "fraud":  {"model_name": "gpt-4.1", "temperature": 0.1},
    "rag":    {"model_name": "gpt-4.1", "temperature": 0.1},
    "tavily": {"model_name": "gpt-4.1", "temperature": 0.1},
    "router": {"model_name": "gpt-4.1", "temperature": 0.1},
    "summary": {"model_name": "gpt-4.1", "temperature": 0.1},
    "generic": {"model_name": "gpt-4.1-mini", "temperature": 0.1},
}

# Cache for LLM instances (singleton pattern)
_llm_cache: Dict[str, "OpenAIChatLLM"] = {}
_langchain_llm_cache: Dict[str, any] = {}

class OpenAIChatLLM:
    def __init__(self, client: OpenAI, model_name: str, config: Optional[dict] = None, max_retries: int = 3, retry_delay: float = 1.0):
        self.client = client
        self.model_name = model_name
        self.config = (config or DEFAULT_MODEL_CONFIG).copy()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _retry_with_backoff(self, func, *args, **kwargs):
        # retry mechanism with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (RateLimitError, APIConnectionError, APITimeoutError, APIError) as e:
                last_exception = e
                # if this is the last attempt, raise the exception
                if attempt == self.max_retries - 1:
                    raise
                # exponential backoff: 1s, 2s, 4s...
                delay = self.retry_delay * (2 ** attempt)
                time.sleep(delay)
            except Exception as e:
                raise
        if last_exception:
            raise last_exception

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format=None,
    ):
        kwargs = self.config.copy()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # if response_format is None, use the regular create() method
        # otherwise use the parse() method for structured parsing
        if response_format is None:
            response = self._retry_with_backoff(
                self.client.chat.completions.create,
                model=self.model_name,
                messages=messages,
                **kwargs,
            )
            return response.choices[0].message.content
        else:
            response = self._retry_with_backoff(
                self.client.beta.chat.completions.parse,
                model=self.model_name,
                messages=messages,
                response_format=response_format,
                **kwargs,
            )
            message = response.choices[0].message
            parsed = getattr(message, "parsed", None)
            if parsed is not None:
                return parsed
            return message.content

def get_llm(
    role: LLMRole = "generic",
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    use_langchain: bool = False,
    **kwargs
) -> Union["OpenAIChatLLM", any]:
    """
    Factory function to get or create an LLM instance based on role.
    Uses singleton pattern to avoid creating duplicate instances.

    Args:
        role: The role/purpose of the LLM (sql, fraud, rag, tavily, router, summary, generic)
        model_name: Override default model name for this role
        temperature: Override default temperature for this role
        use_langchain: If True, returns langchain's ChatOpenAI (for chain compatibility)
        **kwargs: Additional parameters to pass to LLM constructor

    Returns:
        OpenAIChatLLM instance (or ChatOpenAI if use_langchain=True)
    """
    # Get defaults for this role
    defaults = ROLE_DEFAULTS.get(role, ROLE_DEFAULTS["generic"])

    # Use provided values or fall back to role defaults
    final_model_name = model_name or defaults["model_name"]
    final_temperature = temperature if temperature is not None else defaults["temperature"]

    if use_langchain:
        # Return langchain ChatOpenAI for chain compatibility
        from langchain_openai import ChatOpenAI

        # Create cache key for langchain LLM
        cache_key = f"langchain:{role}:{final_model_name}:{final_temperature}"

        if cache_key not in _langchain_llm_cache:
            _langchain_llm_cache[cache_key] = ChatOpenAI(
                model=final_model_name,
                temperature=final_temperature,
                max_retries=kwargs.get("max_retries", 3),
                **{k: v for k, v in kwargs.items() if k != "max_retries"}
            )

        return _langchain_llm_cache[cache_key]
    else:
        # Return custom OpenAIChatLLM
        # Create cache key
        cache_key = f"custom:{role}:{final_model_name}:{final_temperature}"

        if cache_key not in _llm_cache:
            # Create config with temperature
            config = DEFAULT_MODEL_CONFIG.copy()
            config["temperature"] = final_temperature

            _llm_cache[cache_key] = OpenAIChatLLM(
                client=client,
                model_name=final_model_name,
                config=config,
                max_retries=kwargs.get("max_retries", 3),
                retry_delay=kwargs.get("retry_delay", 1.0),
            )

        return _llm_cache[cache_key]
