from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from babeldoc.format.pdf import high_level as babeldoc_high_level
from pdf2zh_next.config.cli_env_model import CLIEnvSettingsModel
from pdf2zh_next.config.main import ConfigManager
from pdf2zh_next.config.model import BasicSettings
from pdf2zh_next.config.model import GUISettings
from pdf2zh_next.config.model import PDFSettings
from pdf2zh_next.config.model import SettingsModel
from pdf2zh_next.config.model import TranslationSettings
from pdf2zh_next.config.translate_engine_model import OpenAISettings
from pdf2zh_next.gui import SaveMode
from pdf2zh_next.gui import _build_cli_settings_from_ui
from pdf2zh_next.gui import _build_translate_settings
from pdf2zh_next.gui import _read_cli_settings_from_toml
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


def test_gui_build_translate_settings_updates_profile_fields(tmp_path: Path):
    prompt_override = tmp_path / "prompt-overrides.yaml"
    prompt_override.write_text(
        PROMPT_OVERRIDE_CONTENT.replace(
            "profiles:\n  default:\n",
            "profiles:\n  custom-prompt-profile:\n",
            1,
        ),
        encoding="utf-8",
    )
    model_param_override = tmp_path / "model-params.yaml"
    model_param_override.write_text(
        MODEL_PARAM_OVERRIDE_CONTENT.replace(
            "profiles:\n  default:\n",
            "profiles:\n  custom-model-profile:\n",
            1,
        ),
        encoding="utf-8",
    )
    input_pdf = tmp_path / "input.pdf"
    input_pdf.write_bytes(b"%PDF-1.4\n")

    base_settings = CLIEnvSettingsModel(
        openai=True,
        openai_detail={"openai_api_key": "test-key"},
    )

    ui_inputs = {
        "service": "OpenAI",
        "lang_from": "English",
        "lang_to": "Simplified Chinese",
        "page_range": "All",
        "page_input": "",
        "ignore_cache": False,
        "no_mono": False,
        "no_dual": False,
        "dual_translate_first": False,
        "use_alternating_pages_dual": False,
        "watermark_output_mode": "Watermarked",
        "rate_limit_mode": "Custom",
        "custom_qps": 4,
        "custom_pool_workers": None,
        "min_text_length": 5,
        "rpc_doclayout": "",
        "prompt_profile_input": "custom-prompt-profile",
        "prompt_override_file_input": str(prompt_override),
        "model_param_profile_input": "custom-model-profile",
        "model_param_override_file_input": str(model_param_override),
        "custom_system_prompt_input": "",
        "glossaries": None,
        "save_auto_extracted_glossary": False,
        "enable_auto_term_extraction": True,
        "primary_font_family": "Auto",
        "skip_clean": False,
        "disable_rich_text_translate": False,
        "enhance_compatibility": False,
        "split_short_lines": False,
        "short_line_split_factor": 0.8,
        "translate_table_text": True,
        "skip_scanned_detection": False,
        "ocr_workaround": False,
        "max_pages_per_part": 0,
        "formular_font_pattern": "",
        "formular_char_pattern": "",
        "auto_enable_ocr_workaround": False,
        "only_include_translated_page": False,
        "merge_alternating_line_numbers": True,
        "remove_non_formula_lines": True,
        "non_formula_line_iou_threshold": 0.9,
        "figure_table_protection_threshold": 0.9,
        "skip_formula_offset_calculation": False,
        "term_service": "Follow main translation engine",
        "term_rate_limit_mode": None,
        "term_rpm_input": None,
        "term_concurrent_threads": None,
        "term_custom_qps": None,
        "term_custom_pool_workers": None,
        "openai_model": "gpt-4o-mini",
        "openai_base_url": None,
        "openai_api_key": "test-key",
        "openai_timeout": None,
        "openai_temperature": None,
        "openai_reasoning_effort": None,
        "openai_enable_json_mode": None,
        "openai_send_temprature": None,
        "openai_send_reasoning_effort": None,
    }

    result = _build_translate_settings(
        base_settings=base_settings,
        file_path=input_pdf,
        output_dir=tmp_path,
        save_mode=SaveMode.never,
        ui_inputs=ui_inputs,
    )

    assert result.translation.prompt_profile == "custom-prompt-profile"
    assert result.translation.prompt_override_file == str(prompt_override)
    assert result.translation.model_param_profile == "custom-model-profile"
    assert result.translation.model_param_override_file == str(model_param_override)


def test_gui_build_cli_settings_keeps_page_selection_for_persistence():
    base_settings = CLIEnvSettingsModel(
        openai=True,
        openai_detail={"openai_api_key": "test-key"},
    )

    ui_inputs = {
        "service": "OpenAI",
        "lang_from": "English",
        "lang_to": "Simplified Chinese",
        "page_range": "Range",
        "page_input": "2-4",
        "ignore_cache": True,
        "no_mono": False,
        "no_dual": False,
        "dual_translate_first": True,
        "use_alternating_pages_dual": False,
        "watermark_output_mode": "No Watermark",
        "rate_limit_mode": "Custom",
        "custom_qps": 9,
        "custom_pool_workers": 12,
        "min_text_length": 5,
        "rpc_doclayout": "",
        "prompt_profile_input": "default",
        "prompt_override_file_input": "",
        "model_param_profile_input": "default",
        "model_param_override_file_input": "",
        "custom_system_prompt_input": "",
        "glossaries": ["unused.csv"],
        "save_auto_extracted_glossary": True,
        "enable_auto_term_extraction": True,
        "primary_font_family": "Auto",
        "skip_clean": False,
        "disable_rich_text_translate": False,
        "enhance_compatibility": False,
        "split_short_lines": False,
        "short_line_split_factor": 0.8,
        "translate_table_text": True,
        "skip_scanned_detection": False,
        "ocr_workaround": False,
        "max_pages_per_part": 0,
        "formular_font_pattern": "",
        "formular_char_pattern": "",
        "auto_enable_ocr_workaround": False,
        "only_include_translated_page": True,
        "merge_alternating_line_numbers": True,
        "remove_non_formula_lines": True,
        "non_formula_line_iou_threshold": 0.9,
        "figure_table_protection_threshold": 0.9,
        "skip_formula_offset_calculation": False,
        "term_service": "Follow main translation engine",
        "term_rate_limit_mode": "Custom",
        "term_rpm_input": None,
        "term_concurrent_threads": None,
        "term_custom_qps": 7,
        "term_custom_pool_workers": 10,
        "openai_model": "gpt-4o-mini",
        "openai_base_url": None,
        "openai_api_key": "test-key",
        "openai_timeout": None,
        "openai_temperature": None,
        "openai_reasoning_effort": None,
        "openai_enable_json_mode": None,
        "openai_send_temprature": None,
        "openai_send_reasoning_effort": None,
    }

    result = _build_cli_settings_from_ui(base_settings, ui_inputs)

    assert result.pdf.pages == "2-4"
    assert result.translation.ignore_cache is True
    assert result.translation.qps == 9
    assert result.translation.pool_max_workers == 12
    assert result.translation.glossaries is None
    assert result.basic.input_files == set()


def test_gui_can_read_uploaded_toml_config(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config = CLIEnvSettingsModel(
        openai=True,
        openai_detail={"openai_api_key": "test-key"},
    )
    ConfigManager()._write_toml_file(config_path, config.model_dump(mode="json"))

    imported = _read_cli_settings_from_toml(str(config_path))

    assert imported.openai is True
    assert imported.openai_detail.openai_api_key == "test-key"
