import time
from typing import Optional, Tuple

from src.settings import settings
from src.utils.logger import logger


class LLMClient:
    """Multi-provider LLM client supporting Ollama, DeepSeek, and DashScope."""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self._check_availability()

    def _check_availability(self):
        if self.provider == "ollama":
            try:
                import requests
                resp = requests.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=5)
                if resp.status_code == 200:
                    models = [m["name"] for m in resp.json().get("models", [])]
                    logger.info(f"Ollama available. Models: {models}")
                else:
                    logger.warning("Ollama API returned non-200")
            except Exception as e:
                logger.warning(f"Ollama not reachable: {e}")

        elif self.provider == "deepseek":
            if not settings.DEEPSEEK_API_KEY:
                logger.warning("DeepSeek API key not set")
            else:
                logger.info("DeepSeek configured")

        elif self.provider == "dashscope":
            if not settings.DASHSCOPE_API_KEY:
                logger.warning("DashScope API key not set")
            else:
                logger.info("DashScope configured")

    def generate_response_with_metrics(self, prompt: str) -> Tuple[str, float, float]:
        """Generate response and return (content, first_token_latency, total_latency)."""
        start_time = time.time()
        first_token_time = None
        content = ""

        try:
            if self.provider == "ollama":
                content = self._generate_ollama(prompt)
            elif self.provider == "deepseek":
                content = self._generate_deepseek(prompt)
            elif self.provider == "dashscope":
                content = self._generate_dashscope(prompt)
            else:
                content = "LLM Service unavailable. Check LLM_PROVIDER config."
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            content = f"Error generating response: {str(e)}"

        total_time = time.time() - start_time
        first_token_latency = total_time  # fallback when no streaming
        return content, first_token_latency, total_time

    def generate_response(
        self, query: str, context: str, system_prompt: Optional[str] = None
    ) -> str:
        """Generate a RAG response based on context."""
        if system_prompt:
            prompt = f"""{system_prompt}

上下文：
{context}

问题：{query}

要求：
1. 基于上下文回答
2. 输出符合指令要求
"""
            return self._call(prompt)

        prompt = f"""基于以下上下文信息，回答问题。

上下文：
{context}

问题：{query}

要求：
1. 基于上下文回答，不添加外部知识
2. 如上下文无相关信息，明确说明"根据提供的信息无法回答"
3. 引用相关段落编号
4. 保持回答准确、简洁

回答："""
        return self._call(prompt)

    def generate_general_response(self, query: str, context: str = "") -> str:
        """Generate response for general chat (no strict RAG constraints)."""
        prompt = f"""You are a helpful assistant.

{context}

User Question: {query}

Answer:"""
        return self._call(prompt)

    def generate_custom_response(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """Generate response with custom prompt."""
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        return self._call(prompt)

    # ── internal call dispatcher ──────────────────────────────

    def _call(self, prompt: str) -> str:
        if self.provider == "ollama":
            return self._generate_ollama(prompt)
        elif self.provider == "deepseek":
            return self._generate_deepseek(prompt)
        elif self.provider == "dashscope":
            return self._generate_dashscope(prompt)
        return "LLM Service unavailable."

    # ── provider implementations ──────────────────────────────

    def _generate_ollama(self, prompt: str) -> str:
        import requests
        try:
            resp = requests.post(
                f"{settings.OLLAMA_HOST}/api/generate",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 1024},
                },
                timeout=120,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "")
            return f"[Ollama error: HTTP {resp.status_code}]"
        except requests.exceptions.Timeout:
            return "[Ollama timeout - model may be loading, retry]"
        except Exception as e:
            return f"[Ollama error: {e}]"

    def _generate_deepseek(self, prompt: str) -> str:
        import requests
        try:
            resp = requests.post(
                f"{settings.DEEPSEEK_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.DEEPSEEK_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 1024,
                },
                timeout=60,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            return f"[DeepSeek error: HTTP {resp.status_code}]"
        except Exception as e:
            return f"[DeepSeek error: {e}]"

    def _generate_dashscope(self, prompt: str) -> str:
        try:
            from langchain_community.chat_models import ChatTongyi
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = ChatTongyi(
                model=settings.LLM_MODEL,
                temperature=0.3,
                api_key=settings.DASHSCOPE_API_KEY,
            )
            messages = [
                SystemMessage(content="You are a helpful RAG assistant."),
                HumanMessage(content=prompt),
            ]
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"DashScope generation failed: {e}")
            return f"[DashScope error: {e}]"
