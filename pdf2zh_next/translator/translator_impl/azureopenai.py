import logging

import openai
from babeldoc.utils.atomic_integer import AtomicInteger
from pdf2zh_next.config.model import SettingsModel
from pdf2zh_next.translator.base_rate_limiter import BaseRateLimiter
from pdf2zh_next.translator.base_translator import BaseTranslator
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

logger = logging.getLogger(__name__)


class AzureOpenAITranslator(BaseTranslator):
    name = "azure-openai"

    def __init__(
        self,
        settings: SettingsModel,
        rate_limiter: BaseRateLimiter,
    ):
        super().__init__(settings, rate_limiter)
        self.model = settings.translate_engine_settings.azure_openai_model
        runtime_params = self.get_runtime_model_params(
            supported_keys={"temperature", "top_p", "max_tokens", "timeout_seconds"},
        )
        client_timeout = runtime_params.pop("timeout_seconds", None)
        self.options = dict(runtime_params)
        self.client = openai.AzureOpenAI(
            azure_endpoint=settings.translate_engine_settings.azure_openai_base_url,
            azure_deployment=self.model,
            api_version=settings.translate_engine_settings.azure_openai_api_version,
            api_key=settings.translate_engine_settings.azure_openai_api_key,
            timeout=client_timeout if client_timeout else openai.NOT_GIVEN,
        )
        self.add_cache_impact_parameters_from_mapping(self.options)
        self.add_cache_impact_parameters("model", self.model)
        self.add_cache_impact_parameters("prompt", self.prompt(""))
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=self.prompt(text),
        )
        self.record_chat_usage(response)
        return self.extract_chat_message_text(response)

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        if text is None:
            return None

        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=self.build_user_message(text),
        )
        self.record_chat_usage(response)
        return self.extract_chat_message_text(response)
