from ace_appworld import config
import logging

logger = logging.getLogger(__name__)

class LLM:
    def __init__(self, model_name: str, model_provider: str):
        self.model_name = model_name
        self.model_provider = model_provider
        self.temperature = config.TEMPERATURE
        self.max_tokens = config.MAX_TOKENS
        # Initialize client based on provider
        if self.model_provider == "openrouter":
            self.api_key = config.OPENROUTER_API_KEY
            self._init_openrouter()
        elif self.model_provider == "gemini":
            self.api_key = config.GEMINI_API_KEY
            self._init_gemini()
        elif self.model_provider == "ollama":
            self.api_key = config.OLLAMA_API_KEY
            self._init_ollama()
        else:
            raise ValueError(f"Unknown provider in config: {self.model_provider}")

    def _init_openrouter(self):
        if not self.api_key:
            raise ValueError("OpenRouter API key not set in .env or config.")
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        logger.debug("Reflector OpenRouter client configured")
    
    def _init_gemini(self):
        try:
            from google import genai
            from google.genai import types
            if not self.api_key:
                raise ValueError("Gemini API key not set in .env or config.")
            os.environ["GOOGLE_API_KEY"] = self.api_key
            self.gemini_client = genai.Client()
            self.gemini_types = types
            logger.debug("Reflector Gemini client initialized")
        except ImportError:
            logger.error("google-genai not installed. Run: uv pip install google-genai")
            raise
    
    def _init_ollama(self):
        if not self.api_key:
            raise ValueError("Ollama Cloud API key not set in .env or config.")
        self.ollama_cloud_url = "https://ollama.com"
        logger.debug("Reflector Ollama client configured")
    
    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM for reflection"""
        try:
            if self.model_provider == "openrouter":
                return self._call_openrouter(prompt)
            elif self.model_provider == "gemini":
                return self._call_gemini(prompt)
            elif self.model_provider == "ollama":
                return self._call_ollama(prompt)
            else:
                raise ValueError(f"Provider {self.model_provider} not supported")
        except Exception as e:
            logger.error(f"Reflector LLM call failed: {e}", exc_info=True)
            raise

    def _call_openrouter(self, prompt: str) -> str:
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens, 
            "provider": {
                "sort": "price"
            }
        }
        response = requests.post(self.openrouter_url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    def _call_gemini(self, prompt: str) -> str:
        config_gemini = self.gemini_types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens
        )
        response = self.gemini_client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config_gemini
        )
        return response.text

    def _call_ollama(self, prompt: str) -> str:
        from ollama import Client
        client = Client(host=self.ollama_cloud_url, headers={'Authorization': f'Bearer {self.api_key}'})
        response = client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature, "num_predict": self.max_tokens, "timeout": 180}
        )
        return response['message']['content']