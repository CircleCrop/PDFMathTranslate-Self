from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from babeldoc.format.pdf import high_level as babeldoc_high_level
from pdf2zh_next.config.model import BasicSettings
from pdf2zh_next.config.model import GUISettings
from pdf2zh_next.config.model import PDFSettings
from pdf2zh_next.config.model import SettingsModel
from pdf2zh_next.config.model import TranslationSettings
from pdf2zh_next.config.translate_engine_model import OpenAISettings
from pdf2zh_next.runtime import ModelFamily
from pdf2zh_next.runtime import build_babeldoc_role_block
from pdf2zh_next.runtime import load_model_param_bundle
from pdf2zh_next.runtime import load_prompt_bundle
from pdf2zh_next.runtime.babeldoc_patch import PatchedAutomaticTermExtractor
from pdf2zh_next.runtime.babeldoc_patch import PatchedILTranslator
from pdf2zh_next.runtime.babeldoc_patch import PatchedILTranslatorLLMOnly
from pdf2zh_next.runtime.babeldoc_patch import apply_babeldoc_runtime_patch
from pdf2zh_next.runtime.babeldoc_patch import attach_babeldoc_runtime_context
from pdf2zh_next.runtime.babeldoc_patch import validate_babeldoc_patch_targets
from pdf2zh_next.translator.base_translator import BaseTranslator

PROMPT_OVERRIDE_CONTENT = """\
profiles:
  default:
    prompts:
      translation.main_role_block_default: |
        OVERRIDE MAIN ROLE $lang_out
      translation.role_block_default: |
        OVERRIDE BABEL ROLE $lang_out
      translation.main_prompt: |
        MAIN OVERRIDE :: $role_block :: $text_to_translate
      translation.paragraph_prompt: |
        PARAGRAPH OVERRIDE :: $role_block :: $text_to_translate
      translation.batch_prompt: |
        BATCH OVERRIDE :: $role_block :: $json_input_str
      term_extraction.extract_terms_prompt: |
        TERM OVERRIDE :: $target_language :: $text_to_process
"""

MODEL_PARAM_OVERRIDE_CONTENT = """\
profiles:
  default:
    defaults:
      temperature: 0.1
      top_p: 0.9
    families:
      gpt:
        temperature: 0.2
      claude:
        max_tokens: 1024
      gemini:
        top_p: 0.7
    providers:
      openai:
        max_tokens: 2048
      claude_code:
        max_turns: 2
"""


class DummyRateLimiter:
    def wait(self, rate_limit_params=None):
        return None


class DummyTranslator(BaseTranslator):
    name = "dummy"

    def __init__(self, settings: SettingsModel):
        super().__init__(settings, DummyRateLimiter())
        self.model = "gpt-4o-mini"

    def do_translate(self, text, rate_limit_params: dict = None):
        return text

    def do_llm_translate(self, text, rate_limit_params: dict = None):
        return text


def _build_settings(
    prompt_override_file: str | None = None,
    model_param_override_file: str | None = None,
    custom_system_prompt: str | None = None,
) -> SettingsModel:
    return SettingsModel(
        basic=BasicSettings(),
        translation=TranslationSettings(
            prompt_override_file=prompt_override_file,
            model_param_override_file=model_param_override_file,
            custom_system_prompt=custom_system_prompt,
        ),
        pdf=PDFSettings(),
        gui_settings=GUISettings(),
        translate_engine_settings=OpenAISettings(openai_api_key="test-key"),
    )


def test_prompt_override_file_updates_main_prompt(tmp_path: Path):
    prompt_override = tmp_path / "prompt-overrides.yaml"
    prompt_override.write_text(PROMPT_OVERRIDE_CONTENT, encoding="utf-8")

    translator = DummyTranslator(_build_settings(prompt_override_file=str(prompt_override)))

    prompt = translator.prompt("Hello world")[0]["content"]
    assert "MAIN OVERRIDE" in prompt
    assert "OVERRIDE MAIN ROLE zh" in prompt
    assert "Hello world" in prompt


def test_custom_system_prompt_overrides_role_block(tmp_path: Path):
    prompt_override = tmp_path / "prompt-overrides.yaml"
    prompt_override.write_text(PROMPT_OVERRIDE_CONTENT, encoding="utf-8")

    translator = DummyTranslator(
        _build_settings(
            prompt_override_file=str(prompt_override),
            custom_system_prompt="SYSTEM ROLE OVERRIDE",
        )
    )

    prompt = translator.prompt("Hello world")[0]["content"]
    assert "SYSTEM ROLE OVERRIDE" in prompt
    assert "OVERRIDE MAIN ROLE" not in prompt


def test_model_param_bundle_merges_family_and_provider_overrides(tmp_path: Path):
    override_file = tmp_path / "model-params.yaml"
    override_file.write_text(MODEL_PARAM_OVERRIDE_CONTENT, encoding="utf-8")

    bundle = load_model_param_bundle(override_file=str(override_file))

    family, params = bundle.resolve("openai", "gpt-4o")
    assert family is ModelFamily.GPT
    assert params["temperature"] == 0.2
    assert params["top_p"] == 0.9
    assert params["max_tokens"] == 2048

    family, params = bundle.resolve("openai", "gemini-2.5-flash")
    assert family is ModelFamily.GEMINI
    assert params["top_p"] == 0.7

    family, params = bundle.resolve("claude_code", None)
    assert family is ModelFamily.CLAUDE
    assert params["max_turns"] == 2
    assert params["max_tokens"] == 1024


def test_babeldoc_role_builder_respects_custom_prompt(tmp_path: Path):
    prompt_override = tmp_path / "prompt-overrides.yaml"
    prompt_override.write_text(PROMPT_OVERRIDE_CONTENT, encoding="utf-8")

    prompt_bundle = load_prompt_bundle(override_file=str(prompt_override))
    role_block = build_babeldoc_role_block(
        prompt_bundle=prompt_bundle,
        lang_out="zh",
        custom_system_prompt="BABELDOC SYSTEM OVERRIDE",
    )

    assert "BABELDOC SYSTEM OVERRIDE" in role_block
    assert "Follow all rules strictly." in role_block
    assert "OVERRIDE BABEL ROLE" not in role_block


def test_prompt_override_file_updates_babeldoc_batch_prompt(tmp_path: Path):
    prompt_override = tmp_path / "prompt-overrides.yaml"
    prompt_override.write_text(PROMPT_OVERRIDE_CONTENT, encoding="utf-8")
    prompt_bundle = load_prompt_bundle(override_file=str(prompt_override))
    model_param_bundle = load_model_param_bundle()

    translation_config = SimpleNamespace(lang_out="zh", custom_system_prompt=None)
    attach_babeldoc_runtime_context(
        translation_config=translation_config,
        prompt_bundle=prompt_bundle,
        model_param_bundle=model_param_bundle,
        model_name="gpt-4o-mini",
    )

    translator = PatchedILTranslatorLLMOnly.__new__(PatchedILTranslatorLLMOnly)
    translator.translation_config = translation_config
    translator._cached_glossaries = []

    prompt = translator._build_llm_prompt("[]", None, None, "")
    assert "BATCH OVERRIDE" in prompt
    assert "OVERRIDE BABEL ROLE zh" in prompt


def test_prompt_override_file_updates_term_extraction_prompt(tmp_path: Path):
    prompt_override = tmp_path / "prompt-overrides.yaml"
    prompt_override.write_text(PROMPT_OVERRIDE_CONTENT, encoding="utf-8")
    prompt_bundle = load_prompt_bundle(override_file=str(prompt_override))
    model_param_bundle = load_model_param_bundle()

    translation_config = SimpleNamespace(
        lang_out="zh",
        raise_if_cancelled=lambda: None,
    )
    attach_babeldoc_runtime_context(
        translation_config=translation_config,
        prompt_bundle=prompt_bundle,
        model_param_bundle=model_param_bundle,
        model_name="gpt-4o-mini",
    )

    captured = {}

    class FakeTracker:
        def append_paragraph_unicode(self, value):
            captured.setdefault("paragraphs", []).append(value)

        def set_input(self, value):
            captured["prompt"] = value

        def set_output(self, value):
            captured["output"] = value

    class FakeSharedContext:
        user_glossaries = []

        def add_raw_extracted_term_pair(self, src, tgt):
            captured.setdefault("terms", []).append((src, tgt))

    class FakeTranslateEngine:
        def llm_translate(self, prompt, rate_limit_params=None):
            captured["rate_limit_params"] = rate_limit_params
            return '[{"src": "LLM", "tgt": "大语言模型"}]'

    extractor = PatchedAutomaticTermExtractor.__new__(PatchedAutomaticTermExtractor)
    extractor.translation_config = translation_config
    extractor.shared_context = FakeSharedContext()
    extractor.translate_engine = FakeTranslateEngine()

    batch = SimpleNamespace(
        paragraphs=[SimpleNamespace(unicode="Large language model")],
        tracker=FakeTracker(),
    )
    extractor.extract_terms_from_paragraphs(batch, pbar=None, paragraph_token_count=7)

    assert "TERM OVERRIDE" in captured["prompt"]
    assert captured["terms"] == [("LLM", "大语言模型")]


def test_babeldoc_patch_targets_are_valid_and_patch_is_applied():
    validate_babeldoc_patch_targets()
    apply_babeldoc_runtime_patch()

    assert babeldoc_high_level.ILTranslator is PatchedILTranslator
    assert babeldoc_high_level.ILTranslatorLLMOnly is PatchedILTranslatorLLMOnly
    assert babeldoc_high_level.AutomaticTermExtractor is PatchedAutomaticTermExtractor
