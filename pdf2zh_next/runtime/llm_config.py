from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from pathlib import Path
from string import Template
from typing import Any

import yaml


class ModelFamily(str, Enum):
    DEFAULT = "default"
    GPT = "gpt"
    CLAUDE = "claude"
    GEMINI = "gemini"


REQUIRED_PROMPT_KEYS = {
    "translation.main_role_block_default",
    "translation.role_block_default",
    "translation.main_prompt",
    "translation.paragraph_prompt",
    "translation.batch_prompt",
    "term_extraction.extract_terms_prompt",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_top_level_mapping(data: Any, source: str) -> dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{source} must contain a mapping at top level")
    return data


def _load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML file: {path}") from exc
    return _normalize_top_level_mapping(data, f"YAML file {path}")


def _load_packaged_yaml(*parts: str) -> dict[str, Any]:
    package_root = resources.files("pdf2zh_next")
    resource = package_root.joinpath(*parts)
    data = yaml.safe_load(resource.read_text(encoding="utf-8"))
    return _normalize_top_level_mapping(data, f"Packaged YAML {'/'.join(parts)}")


def _resolve_override_file(path: str | None) -> Path | None:
    if not path:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved


@dataclass(frozen=True)
class PromptBundle:
    profile_name: str
    prompts: dict[str, str]

    def get(self, key: str) -> str:
        try:
            return self.prompts[key]
        except KeyError as exc:
            raise KeyError(f"Prompt template not found: {key}") from exc

    def render(self, key: str, **values: Any) -> str:
        template = Template(self.get(key))
        return template.substitute(values)


@dataclass(frozen=True)
class ModelParamBundle:
    profile_name: str
    defaults: dict[str, Any]
    families: dict[str, dict[str, Any]]
    providers: dict[str, dict[str, Any]]

    def resolve(
        self,
        provider_name: str,
        model_name: str | None,
    ) -> tuple[ModelFamily, dict[str, Any]]:
        family = detect_model_family(model_name=model_name, provider_name=provider_name)
        resolved: dict[str, Any] = {}
        family_params = self.families.get(family.value, {})
        resolved.update(self.defaults)
        resolved.update(family_params)
        resolved.update(self.providers.get(provider_name, {}))
        # Gemini currently uses an OpenAI-compatible transport path in this project.
        # Re-apply Gemini family overrides so Gemini-specific controls still win over
        # generic OpenAI provider defaults such as max_tokens.
        if provider_name == "openai" and family is ModelFamily.GEMINI:
            resolved.update(family_params)
        resolved = {k: v for k, v in resolved.items() if v is not None}
        return family, resolved


def _set_override_if_present(
    target: dict[str, Any],
    key: str,
    value: Any,
) -> None:
    if value is not None:
        target[key] = value


def _extract_profile(config: dict[str, Any], profile_name: str) -> dict[str, Any]:
    profiles = config.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("Config must contain a 'profiles' mapping")
    try:
        profile = profiles[profile_name]
    except KeyError as exc:
        raise ValueError(f"Profile not found: {profile_name}") from exc
    if not isinstance(profile, dict):
        raise ValueError(f"Profile must be a mapping: {profile_name}")
    return profile


def _normalize_named_mapping(
    values: Any,
    *,
    section_name: str,
    profile_name: str,
) -> dict[str, dict[str, Any]]:
    if not values:
        return {}
    if not isinstance(values, dict):
        raise ValueError(f"{section_name} must be a mapping: {profile_name}")

    normalized: dict[str, dict[str, Any]] = {}
    for key, value in values.items():
        if not isinstance(value, dict):
            raise ValueError(f"{section_name} entry must be a mapping: {key}")
        normalized[key] = value
    return normalized


def load_prompt_bundle(
    profile_name: str = "default",
    override_file: str | None = None,
) -> PromptBundle:
    config = _load_packaged_yaml("prompts", "default.yaml")
    override_path = _resolve_override_file(override_file)
    if override_path:
        config = _deep_merge(config, _load_yaml_file(override_path))

    profile = _extract_profile(config, profile_name)
    prompts = profile.get("prompts")
    if not isinstance(prompts, dict):
        raise ValueError(f"Prompt profile must contain a 'prompts' mapping: {profile_name}")

    missing_keys = REQUIRED_PROMPT_KEYS - set(prompts)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(
            f"Prompt profile '{profile_name}' is missing required templates: {missing}"
        )

    normalized_prompts = {}
    for key, value in prompts.items():
        if not isinstance(value, str):
            raise ValueError(f"Prompt template must be a string: {key}")
        normalized_prompts[key] = value

    return PromptBundle(profile_name=profile_name, prompts=normalized_prompts)


def load_model_param_bundle(
    profile_name: str = "default",
    override_file: str | None = None,
) -> ModelParamBundle:
    config = _load_packaged_yaml("model_params", "default.yaml")
    override_path = _resolve_override_file(override_file)
    if override_path:
        config = _deep_merge(config, _load_yaml_file(override_path))

    profile = _extract_profile(config, profile_name)
    defaults = profile.get("defaults") or {}
    families = profile.get("families") or {}
    providers = profile.get("providers") or {}

    if not isinstance(defaults, dict):
        raise ValueError(f"Model param profile defaults must be a mapping: {profile_name}")

    return ModelParamBundle(
        profile_name=profile_name,
        defaults=defaults,
        families=_normalize_named_mapping(
            families,
            section_name="Model param family",
            profile_name=profile_name,
        ),
        providers=_normalize_named_mapping(
            providers,
            section_name="Model param provider",
            profile_name=profile_name,
        ),
    )


def apply_translation_model_param_overrides(
    bundle: ModelParamBundle,
    translation_settings: Any,
) -> ModelParamBundle:
    defaults = dict(bundle.defaults)
    families = {key: dict(value) for key, value in bundle.families.items()}
    providers = {key: dict(value) for key, value in bundle.providers.items()}

    _set_override_if_present(defaults, "temperature", translation_settings.llm_temperature)
    _set_override_if_present(defaults, "top_p", translation_settings.llm_top_p)
    _set_override_if_present(defaults, "top_k", translation_settings.llm_top_k)
    _set_override_if_present(defaults, "max_tokens", translation_settings.llm_max_tokens)
    _set_override_if_present(
        defaults,
        "timeout_seconds",
        translation_settings.llm_timeout_seconds,
    )

    babeldoc_params = dict(providers.get("babeldoc", {}))
    _set_override_if_present(
        babeldoc_params,
        "paragraph_batch_token_limit",
        translation_settings.paragraph_batch_token_limit,
    )
    _set_override_if_present(
        babeldoc_params,
        "paragraph_batch_size_limit",
        translation_settings.paragraph_batch_size_limit,
    )
    _set_override_if_present(
        babeldoc_params,
        "term_batch_token_limit",
        translation_settings.term_batch_token_limit,
    )
    _set_override_if_present(
        babeldoc_params,
        "term_batch_size_limit",
        translation_settings.term_batch_size_limit,
    )
    _set_override_if_present(
        babeldoc_params,
        "llm_output_ratio_min",
        translation_settings.llm_output_ratio_min,
    )
    _set_override_if_present(
        babeldoc_params,
        "llm_output_ratio_max",
        translation_settings.llm_output_ratio_max,
    )
    _set_override_if_present(
        babeldoc_params,
        "same_as_input_min_input_tokens",
        translation_settings.same_as_input_min_input_tokens,
    )
    _set_override_if_present(
        babeldoc_params,
        "same_text_edit_distance_threshold",
        translation_settings.same_text_edit_distance_threshold,
    )
    _set_override_if_present(
        babeldoc_params,
        "same_text_min_input_tokens",
        translation_settings.same_text_min_input_tokens,
    )
    if babeldoc_params:
        providers["babeldoc"] = babeldoc_params

    openai_params = dict(providers.get("openai", {}))
    _set_override_if_present(
        openai_params,
        "max_tokens",
        translation_settings.openai_max_tokens,
    )
    if openai_params:
        providers["openai"] = openai_params

    gemini_params = dict(families.get(ModelFamily.GEMINI.value, {}))
    _set_override_if_present(
        gemini_params,
        "temperature",
        translation_settings.gemini_temperature,
    )
    _set_override_if_present(gemini_params, "top_p", translation_settings.gemini_top_p)
    _set_override_if_present(
        gemini_params,
        "max_tokens",
        translation_settings.gemini_max_tokens,
    )
    _set_override_if_present(
        gemini_params,
        "timeout_seconds",
        translation_settings.gemini_timeout_seconds,
    )
    if gemini_params:
        families[ModelFamily.GEMINI.value] = gemini_params

    return ModelParamBundle(
        profile_name=bundle.profile_name,
        defaults=defaults,
        families=families,
        providers=providers,
    )


def detect_model_family(
    model_name: str | None,
    provider_name: str | None = None,
) -> ModelFamily:
    provider = (provider_name or "").lower()
    model = (model_name or "").lower()

    if provider in {"claude_code", "claudecode"} or "claude" in model:
        return ModelFamily.CLAUDE
    if "gemini" in model:
        return ModelFamily.GEMINI
    if re.search(r"\b(gpt|o1|o3|o4)\b", model):
        return ModelFamily.GPT
    return ModelFamily.DEFAULT


def build_main_role_block(
    prompt_bundle: PromptBundle,
    lang_out: str,
    custom_system_prompt: str | None,
) -> str:
    if custom_system_prompt:
        return custom_system_prompt.strip()
    return prompt_bundle.render(
        "translation.main_role_block_default",
        lang_out=lang_out,
    ).strip()


def build_babeldoc_role_block(
    prompt_bundle: PromptBundle,
    lang_out: str,
    custom_system_prompt: str | None,
) -> str:
    if custom_system_prompt:
        role_block = custom_system_prompt.strip()
    else:
        role_block = prompt_bundle.render(
            "translation.role_block_default",
            lang_out=lang_out,
        ).strip()

    if "Follow all rules strictly." not in role_block:
        if not role_block.endswith("\n"):
            role_block += "\n"
        role_block += "Follow all rules strictly."
    return role_block
