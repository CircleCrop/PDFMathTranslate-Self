import contextlib
import logging
import re
from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

from pdf2zh_next.config.model import SettingsModel
from pdf2zh_next.runtime import apply_translation_model_param_overrides
from pdf2zh_next.runtime import build_main_role_block
from pdf2zh_next.runtime import load_model_param_bundle
from pdf2zh_next.runtime import load_prompt_bundle
from pdf2zh_next.translator.base_rate_limiter import BaseRateLimiter
from pdf2zh_next.translator.cache import TranslationCache

logger = logging.getLogger(__name__)


class BaseTranslator(ABC):
    # Due to cache limitations, name should be within 20 characters.
    # cache.py: translate_engine = CharField(max_length=20)
    """translator 的基类，所有的 translator 的实现都需要继承"""

    name = "base"
    lang_map = {}

    def __init__(
        self,
        settings: SettingsModel,
        rate_limiter: BaseRateLimiter,
    ):
        """
        translator class initialization
        :param settings: runtime setting and configuration
        :param rate_limiter: LLM request rate control
        :return: None
        """
        self.settings = settings
        self.prompt_bundle = load_prompt_bundle(
            profile_name=settings.translation.prompt_profile,
            override_file=settings.translation.prompt_override_file,
        )
        self.model_param_bundle = apply_translation_model_param_overrides(
            load_model_param_bundle(
                profile_name=settings.translation.model_param_profile,
                override_file=settings.translation.model_param_override_file,
            ),
            settings.translation,
        )
        self._warned_ignored_runtime_params: set[tuple[str, tuple[str, ...]]] = set()
        self.ignore_cache = settings.translation.ignore_cache
        lang_in = self.lang_map.get(
            settings.translation.lang_in.lower(), settings.translation.lang_in
        )
        lang_out = self.lang_map.get(
            settings.translation.lang_out.lower(), settings.translation.lang_out
        )
        self.lang_in = lang_in
        self.lang_out = lang_out
        self.rate_limiter = rate_limiter

        self.cache = TranslationCache(
            self.name,
            {
                "lang_in": lang_in,
                "lang_out": lang_out,
            },
        )

        self.translate_call_count = 0
        self.translate_cache_call_count = 0

    def __del__(self):
        with contextlib.suppress(Exception):
            logger.info(
                f"{self.name} translate call count: {self.translate_call_count}"
            )
            logger.info(
                f"{self.name} translate cache call count: {self.translate_cache_call_count}",
            )

    def get_runtime_provider_name(self) -> str:
        provider_name_map = {
            "azure-openai": "azure_openai",
            "claudecode": "claude_code",
            "qwen-mt": "qwenmt",
        }
        return provider_name_map.get(self.name, self.name.replace("-", "_"))

    def get_runtime_model_params(
        self,
        supported_keys: set[str] | None = None,
        explicit_values: dict[str, Any] | None = None,
        provider_name: str | None = None,
        model_name: str | None = None,
        warn_ignored: bool = True,
    ) -> dict[str, Any]:
        provider = provider_name or self.get_runtime_provider_name()
        _, resolved = self.model_param_bundle.resolve(
            provider_name=provider,
            model_name=model_name or getattr(self, "model", None),
        )

        merged = dict(resolved)
        for key, value in (explicit_values or {}).items():
            if value is not None:
                merged[key] = value

        if supported_keys is None:
            return merged

        applied = {}
        ignored_keys = []
        for key, value in merged.items():
            if key in supported_keys:
                applied[key] = value
            else:
                ignored_keys.append(key)

        if warn_ignored and ignored_keys:
            self._warn_ignored_runtime_params(provider, ignored_keys)

        return applied

    def _warn_ignored_runtime_params(
        self,
        provider_name: str,
        ignored_keys: list[str],
    ) -> None:
        ignored_key_tuple = tuple(sorted(set(ignored_keys)))
        warning_key = (provider_name, ignored_key_tuple)
        if warning_key in self._warned_ignored_runtime_params:
            return

        self._warned_ignored_runtime_params.add(warning_key)
        logger.warning(
            "Ignoring unsupported runtime model parameters for %s: %s",
            provider_name,
            ", ".join(ignored_key_tuple),
        )

    def add_cache_impact_parameters(self, k: str, v):
        """
        Add parameters that affect the translation quality to distinguish the translation effects under different parameters.
        :param k: key
        :param v: value
        """
        self.cache.add_params(k, v)

    def add_cache_impact_parameters_from_mapping(
        self,
        params: Mapping[str, Any],
    ) -> None:
        for key, value in params.items():
            self.add_cache_impact_parameters(key, value)

    def build_user_message(self, text: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": text}]

    def record_chat_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if not usage:
            return

        self._increment_usage_counter("token_count", getattr(usage, "total_tokens", None))
        self._increment_usage_counter(
            "prompt_token_count",
            getattr(usage, "prompt_tokens", None),
        )
        self._increment_usage_counter(
            "completion_token_count",
            getattr(usage, "completion_tokens", None),
        )

        cached_tokens = getattr(usage, "prompt_cache_hit_tokens", None)
        if cached_tokens is None:
            prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
            cached_tokens = getattr(prompt_tokens_details, "cached_tokens", None)
        self._increment_usage_counter("cache_hit_prompt_token_count", cached_tokens)

    def _increment_usage_counter(self, attr_name: str, value: Any) -> None:
        if value is None:
            return
        counter = getattr(self, attr_name, None)
        if counter is None:
            return
        try:
            counter.inc(int(value))
        except Exception:
            logger.debug("Failed to update usage counter %s", attr_name, exc_info=True)

    def extract_chat_message_text(self, response: Any) -> str:
        content = response.choices[0].message.content
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(self._extract_content_part_text(part) for part in content)
        else:
            text = "" if content is None else str(content)
        return self._remove_cot_content(text.strip())

    def _extract_content_part_text(self, part: Any) -> str:
        if isinstance(part, str):
            return part
        if isinstance(part, dict):
            text = part.get("text")
            return text if isinstance(text, str) else ""
        text = getattr(part, "text", None)
        return text if isinstance(text, str) else ""

    def translate(self, text, ignore_cache=False, rate_limit_params: dict = None):
        """
        Translate the text, and the other part should call this method.
        :param text: text to translate
        :return: translated text
        """
        self.translate_call_count += 1
        if not (self.ignore_cache or ignore_cache):
            try:
                cache = self.cache.get(text)
                if cache is not None:
                    self.translate_cache_call_count += 1
                    return cache
            except Exception as e:
                logger.debug(f"try get cache failed, ignore it: {e}")
        self.rate_limiter.wait(rate_limit_params)
        translation = self.do_translate(text, rate_limit_params)
        if not (self.ignore_cache or ignore_cache):
            self.cache.set(text, translation)
        return translation

    def llm_translate(self, text, ignore_cache=False, rate_limit_params: dict = None):
        """
        Translate the text, and the other part should call this method.
        :param text: text to translate
        :return: translated text
        """
        self.translate_call_count += 1
        if not (self.ignore_cache or ignore_cache):
            try:
                cache = self.cache.get(text)
                if cache is not None:
                    self.translate_cache_call_count += 1
                    return cache
            except Exception as e:
                logger.debug(f"try get cache failed, ignore it: {e}")
        self.rate_limiter.wait(rate_limit_params)
        translation = self.do_llm_translate(text, rate_limit_params)
        if not (self.ignore_cache or ignore_cache):
            self.cache.set(text, translation)
        return translation

    def do_llm_translate(self, text, rate_limit_params: dict = None):
        """
        Actual translate text, override this method
        :param text: text to translate
        :return: translated text
        """
        raise NotImplementedError

    @abstractmethod
    def do_translate(self, text, rate_limit_params: dict = None):
        """
        Actual translate text, override this method
        :param text: text to translate
        :return: translated text
        """
        logger.critical(
            f"Do not call BaseTranslator.do_translate. "
            f"Translator: {self}. "
            f"Text: {text}. ",
        )
        raise NotImplementedError

    def _remove_cot_content(self, content: str) -> str:
        """Remove text content with the thought chain from the chat response

        :param content: Non-streaming text content
        :return: Text without a thought chain
        """
        return re.sub(r"^<think>.+?</think>", "", content, count=1, flags=re.DOTALL)

    def __str__(self):
        """
        get translator's info
        """
        return f"{self.name} {self.lang_in} {self.lang_out} {self.model}"

    def get_formular_placeholder(self, placeholder_id: int):
        """
        get formular placeholder
        LLM translator use placeholder to skip the formular char
        :param placeholder_id: placeholder id
        :return formated placeholder and regex placeholder
        """
        return "{v" + str(placeholder_id) + "}", f"{{\\s*v\\s*{placeholder_id}\\s*}}"

    def get_rich_text_left_placeholder(self, placeholder_id: int):
        """
        get rich text placeholder
        :param placeholder_id: placeholder id
        :return the start label of rich text and regex start label
        """
        return (
            f"<style id='{placeholder_id}'>",
            f"<\\s*style\\s*id\\s*=\\s*'\\s*{placeholder_id}\\s*'\\s*>",
        )

    def get_rich_text_right_placeholder(self, placeholder_id: int):
        """
        get rich text placeholder
        :return the end label of rich text and regex end label
        """
        return "</style>", r"<\s*\/\s*style\s*>"

    def prompt(self, text):
        """
        concatent the prompt
        :param text: input text
        :return: the whole prompt for LLM translator
        """
        content = self.render_main_prompt(text)
        return [
            {
                "role": "user",
                "content": content,
            },
        ]

    def render_main_prompt(self, text: str) -> str:
        """Render the main translation prompt for a plain-text translation request."""
        role_block = build_main_role_block(
            prompt_bundle=self.prompt_bundle,
            lang_out=self.lang_out,
            custom_system_prompt=self.settings.translation.custom_system_prompt,
        )
        return self.prompt_bundle.render(
            "translation.main_prompt",
            role_block=role_block,
            lang_out=self.lang_out,
            text_to_translate=text or "",
        )
