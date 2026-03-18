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


class SiliconFlowTranslator(BaseTranslator):
    # https://github.com/openai/openai-python
    name = "siliconflow"

    def __init__(
        self,
        settings: SettingsModel,
        rate_limiter: BaseRateLimiter,
    ):
        super().__init__(settings, rate_limiter)
        self.model = settings.translate_engine_settings.siliconflow_model
        runtime_params = self.get_runtime_model_params(
            supported_keys={"temperature", "top_p", "max_tokens", "timeout_seconds"},
        )
        client_timeout = runtime_params.pop("timeout_seconds", None)
        self.options = dict(runtime_params)
        self.client = openai.OpenAI(
            base_url=settings.translate_engine_settings.siliconflow_base_url,
            api_key=settings.translate_engine_settings.siliconflow_api_key,
            timeout=client_timeout if client_timeout else openai.NOT_GIVEN,
        )
        self.add_cache_impact_parameters_from_mapping(self.options)
        self.enable_thinking = (
            settings.translate_engine_settings.siliconflow_enable_thinking
        )
        self.send_enable_thinking_param = (
            settings.translate_engine_settings.siliconflow_send_enable_thinking_param
        )
        self.add_cache_impact_parameters("enable_thinking", self.enable_thinking)
        self.add_cache_impact_parameters(
            "send_enable_thinking_param", self.send_enable_thinking_param
        )
        self.add_cache_impact_parameters("model", self.model)
        self.add_cache_impact_parameters("prompt", self.prompt(""))
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()
        self.cache_hit_prompt_token_count = AtomicInteger()

        self.enable_json_mode = (
            settings.translate_engine_settings.siliconflow_enable_json_mode
        )
        if self.enable_json_mode:
            self.add_cache_impact_parameters("enable_json_mode", self.enable_json_mode)

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        extra_body = {}

        if self.enable_json_mode:
            extra_body["response_format"] = {"type": "json_object"}
        if self.send_enable_thinking_param:
            extra_body["enable_thinking"] = self.enable_thinking

        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=self.prompt(text),
            extra_body=extra_body,
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

        extra_body = {}

        if self.enable_json_mode:
            extra_body["response_format"] = {"type": "json_object"}
        if self.send_enable_thinking_param:
            extra_body["enable_thinking"] = self.enable_thinking

        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=self.build_user_message(text),
            extra_body=extra_body,
        )
        self.record_chat_usage(response)
        return self.extract_chat_message_text(response)
