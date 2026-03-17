from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from pathlib import Path
from string import Template
from typing import Any

import yaml

logger = logging.getLogger(__name__)


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


def _load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML file: {path}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping at top level: {path}")
    return data


def _load_packaged_yaml(*parts: str) -> dict[str, Any]:
    package_root = resources.files("pdf2zh_next")
    resource = package_root.joinpath(*parts)
    data = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Packaged YAML must contain a mapping: {'/'.join(parts)}")
    return data


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
        resolved.update(self.defaults)
        resolved.update(self.families.get(family.value, {}))
        resolved.update(self.providers.get(provider_name, {}))
        resolved = {k: v for k, v in resolved.items() if v is not None}
        return family, resolved


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
    if not isinstance(families, dict):
        raise ValueError(f"Model param families must be a mapping: {profile_name}")
    if not isinstance(providers, dict):
        raise ValueError(f"Model param providers must be a mapping: {profile_name}")

    normalized_families: dict[str, dict[str, Any]] = {}
    for key, value in families.items():
        if not isinstance(value, dict):
            raise ValueError(f"Model param family entry must be a mapping: {key}")
        normalized_families[key] = value

    normalized_providers: dict[str, dict[str, Any]] = {}
    for key, value in providers.items():
        if not isinstance(value, dict):
            raise ValueError(f"Model param provider entry must be a mapping: {key}")
        normalized_providers[key] = value

    return ModelParamBundle(
        profile_name=profile_name,
        defaults=defaults,
        families=normalized_families,
        providers=normalized_providers,
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
