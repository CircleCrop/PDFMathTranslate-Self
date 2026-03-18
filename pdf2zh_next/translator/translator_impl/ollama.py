import logging

import ollama
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


class OllamaTranslator(BaseTranslator):
    # https://github.com/ollama/ollama
    name = "ollama"

    def __init__(
        self,
        settings: SettingsModel,
        rate_limiter: BaseRateLimiter,
    ):
        super().__init__(settings, rate_limiter)
        self.model = settings.translate_engine_settings.ollama_model
        explicit_runtime_params = {}
        if "num_predict" in settings.translate_engine_settings.model_fields_set:
            explicit_runtime_params["num_predict"] = (
                settings.translate_engine_settings.num_predict
            )
        runtime_params = self.get_runtime_model_params(
            supported_keys={
                "temperature",
                "top_k",
                "top_p",
                "num_predict",
                "num_predict_char_multiplier",
            },
            explicit_values=explicit_runtime_params,
        )
        self.base_options = {
            key: value
            for key, value in runtime_params.items()
            if key in {"temperature", "top_k", "top_p"}
        }
        base_num_predict = runtime_params.get("num_predict")
        self.base_num_predict = (
            int(base_num_predict) if base_num_predict is not None else 0
        )
        char_multiplier = runtime_params.get("num_predict_char_multiplier")
        self.num_predict_char_multiplier = (
            int(char_multiplier) if char_multiplier is not None else 0
        )
        self.client = ollama.Client(
            host=settings.translate_engine_settings.ollama_host,
        )
        self.add_cache_impact_parameters_from_mapping(self.base_options)
        self.add_cache_impact_parameters("num_predict", self.base_num_predict)
        self.add_cache_impact_parameters(
            "num_predict_char_multiplier",
            self.num_predict_char_multiplier,
        )
        self.add_cache_impact_parameters("model", self.model)
        self.add_cache_impact_parameters("prompt", self.prompt(""))
        self.token_count = AtomicInteger()
        self.prompt_token_count = AtomicInteger()
        self.completion_token_count = AtomicInteger()

    def _build_request_options(self, text: str) -> dict:
        options = dict(self.base_options)
        effective_num_predict = self.base_num_predict
        if self.num_predict_char_multiplier > 0:
            effective_num_predict = max(
                effective_num_predict,
                len(text) * self.num_predict_char_multiplier,
            )
        if effective_num_predict > 0:
            options["num_predict"] = effective_num_predict
        return options

    @retry(
        retry=retry_if_exception_type(ollama.ResponseError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        options = self._build_request_options(text)
        response = self.client.chat(
            model=self.model,
            options=options,
            messages=self.prompt(text),
        )
        self.token_count.inc(response.prompt_eval_count + response.eval_count)
        self.prompt_token_count.inc(response.prompt_eval_count)
        self.completion_token_count.inc(response.eval_count)
        message = response.message.content.strip()
        message = self._remove_cot_content(message)
        return message

    @retry(
        retry=retry_if_exception_type(ollama.ResponseError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def do_llm_translate(self, text, rate_limit_params: dict = None):
        if text is None:
            return None

        options = self._build_request_options(text)
        response = self.client.chat(
            model=self.model,
            options=options,
            messages=self.build_user_message(text),
        )
        self.token_count.inc(response.prompt_eval_count + response.eval_count)
        self.prompt_token_count.inc(response.prompt_eval_count)
        self.completion_token_count.inc(response.eval_count)
        message = response.message.content.strip()
        message = self._remove_cot_content(message)
        return message
