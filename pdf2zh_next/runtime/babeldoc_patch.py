from __future__ import annotations

import copy
import inspect
import json
import logging
import re
from typing import Any

import Levenshtein
from babeldoc import __version__ as babeldoc_version
from babeldoc.format.pdf import high_level as babeldoc_high_level
from babeldoc.format.pdf.document_il import Page
from babeldoc.format.pdf.document_il import PdfFont
from babeldoc.format.pdf.document_il import PdfParagraph
from babeldoc.format.pdf.document_il.midend import (
    automatic_term_extractor as automatic_term_extractor_module,
)
from babeldoc.format.pdf.document_il.midend import il_translator as il_translator_module
from babeldoc.format.pdf.document_il.midend import (
    il_translator_llm_only as il_translator_llm_only_module,
)
from babeldoc.format.pdf.document_il.midend.automatic_term_extractor import (
    AutomaticTermExtractor as BabelDOCAutomaticTermExtractor,
)
from babeldoc.format.pdf.document_il.midend.automatic_term_extractor import (
    BatchParagraph as TermBatchParagraph,
)
from babeldoc.format.pdf.document_il.midend.automatic_term_extractor import (
    PageTermExtractTracker,
)
from babeldoc.format.pdf.document_il.midend.il_translator import (
    ILTranslator as BabelDOCILTranslator,
)
from babeldoc.format.pdf.document_il.midend.il_translator_llm_only import (
    BatchParagraph as LLMBatchParagraph,
)
from babeldoc.format.pdf.document_il.midend.il_translator_llm_only import (
    ILTranslatorLLMOnly as BabelDOCILTranslatorLLMOnly,
)
from babeldoc.format.pdf.document_il.utils.paragraph_helper import is_cid_paragraph
from babeldoc.format.pdf.document_il.utils.paragraph_helper import (
    is_placeholder_only_paragraph,
)
from babeldoc.format.pdf.document_il.utils.paragraph_helper import (
    is_pure_numeric_paragraph,
)

from pdf2zh_next.runtime.llm_config import ModelParamBundle
from pdf2zh_next.runtime.llm_config import PromptBundle
from pdf2zh_next.runtime.llm_config import build_babeldoc_role_block

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False

_PROMPT_BUNDLE_ATTR = "_pdf2zh_next_prompt_bundle"
_BABELDOC_PARAMS_ATTR = "_pdf2zh_next_babeldoc_params"

SUPPORTED_BABELDOC_PARAM_KEYS = {
    "paragraph_batch_token_limit",
    "paragraph_batch_size_limit",
    "term_batch_token_limit",
    "term_batch_size_limit",
    "llm_output_ratio_min",
    "llm_output_ratio_max",
    "same_as_input_min_input_tokens",
    "same_text_edit_distance_threshold",
    "same_text_min_input_tokens",
}


def _validate_babeldoc_version() -> None:
    if not babeldoc_version.startswith("0.5."):
        raise RuntimeError(
            "Unsupported BabelDOC version for runtime patching: "
            f"{babeldoc_version}. Only BabelDOC 0.5.x is supported."
        )


def _validate_signature(
    obj: type,
    method_name: str,
    expected_parameters: list[str],
) -> None:
    method = getattr(obj, method_name, None)
    if method is None:
        raise RuntimeError(
            f"BabelDOC runtime patch target is missing: {obj.__name__}.{method_name}"
        )

    actual_parameters = list(inspect.signature(method).parameters)
    if actual_parameters[: len(expected_parameters)] != expected_parameters:
        raise RuntimeError(
            "BabelDOC runtime patch target signature changed for "
            f"{obj.__name__}.{method_name}: expected prefix "
            f"{expected_parameters}, got {actual_parameters}"
        )


def validate_babeldoc_patch_targets() -> None:
    _validate_babeldoc_version()

    _validate_signature(BabelDOCILTranslator, "_build_role_block", ["self"])
    _validate_signature(
        BabelDOCILTranslator,
        "_build_context_block",
        ["self", "title_paragraph", "local_title_paragraph", "translate_input"],
    )
    _validate_signature(
        BabelDOCILTranslator,
        "_build_glossary_block",
        ["self", "text"],
    )
    _validate_signature(
        BabelDOCILTranslator,
        "generate_prompt_for_llm",
        ["self", "text", "title_paragraph", "local_title_paragraph", "translate_input"],
    )

    _validate_signature(
        BabelDOCILTranslatorLLMOnly,
        "process_page",
        ["self", "page", "executor", "pbar", "tracker", "executor2", "translated_ids"],
    )
    _validate_signature(
        BabelDOCILTranslatorLLMOnly,
        "translate_paragraph",
        [
            "self",
            "batch_paragraph",
            "pbar",
            "page_font_map",
            "xobj_font_map",
            "title_paragraph",
            "local_title_paragraph",
            "executor",
            "paragraph_token_count",
            "mp_id",
        ],
    )
    _validate_signature(
        BabelDOCILTranslatorLLMOnly,
        "_build_llm_prompt",
        [
            "self",
            "json_input_str",
            "title_paragraph",
            "local_title_paragraph",
            "batch_text_for_glossary_matching",
        ],
    )

    _validate_signature(
        BabelDOCAutomaticTermExtractor,
        "process_page",
        ["self", "page", "executor", "pbar", "tracker"],
    )
    _validate_signature(
        BabelDOCAutomaticTermExtractor,
        "extract_terms_from_paragraphs",
        ["self", "paragraphs", "pbar", "paragraph_token_count"],
    )

    if not hasattr(babeldoc_high_level, "ILTranslator"):
        raise RuntimeError("BabelDOC high_level.ILTranslator is missing")
    if not hasattr(babeldoc_high_level, "ILTranslatorLLMOnly"):
        raise RuntimeError("BabelDOC high_level.ILTranslatorLLMOnly is missing")
    if not hasattr(babeldoc_high_level, "AutomaticTermExtractor"):
        raise RuntimeError("BabelDOC high_level.AutomaticTermExtractor is missing")


def attach_babeldoc_runtime_context(
    translation_config: Any,
    prompt_bundle: PromptBundle,
    model_param_bundle: ModelParamBundle,
    model_name: str | None,
) -> None:
    _, resolved_params = model_param_bundle.resolve("babeldoc", model_name=model_name)
    filtered_params = {
        key: value
        for key, value in resolved_params.items()
        if key in SUPPORTED_BABELDOC_PARAM_KEYS
    }
    setattr(translation_config, _PROMPT_BUNDLE_ATTR, prompt_bundle)
    setattr(translation_config, _BABELDOC_PARAMS_ATTR, filtered_params)


def _get_prompt_bundle(translation_config: Any) -> PromptBundle:
    prompt_bundle = getattr(translation_config, _PROMPT_BUNDLE_ATTR, None)
    if prompt_bundle is None:
        raise RuntimeError(
            "BabelDOC runtime patch is missing prompt bundle context. "
            "Call attach_babeldoc_runtime_context() before translation."
        )
    return prompt_bundle


def _get_babeldoc_params(translation_config: Any) -> dict[str, Any]:
    return getattr(translation_config, _BABELDOC_PARAMS_ATTR, {})


def _get_babeldoc_param(
    translation_config: Any,
    key: str,
    default: Any,
) -> Any:
    return _get_babeldoc_params(translation_config).get(key, default)


def _advance_progress(pbar, amount: int = 1) -> None:
    if pbar:
        pbar.advance(amount)


def _should_skip_paragraph(
    paragraph: PdfParagraph,
    *,
    translated_ids: set[int] | None = None,
    min_text_length: int | None = None,
) -> bool:
    if paragraph.debug_id is None or paragraph.unicode is None:
        return True
    if translated_ids is not None and id(paragraph) in translated_ids:
        return True
    if is_cid_paragraph(paragraph):
        return True
    if min_text_length is not None and len(paragraph.unicode) < min_text_length:
        return True
    if is_pure_numeric_paragraph(paragraph):
        return True
    if is_placeholder_only_paragraph(paragraph):
        return True
    return False


def _flush_batch_if_needed(
    *,
    items: list[Any],
    total_token_count: int,
    token_limit: int,
    size_limit: int,
    submit_batch,
) -> tuple[list[Any], int]:
    if total_token_count > token_limit or len(items) > size_limit:
        submit_batch(items, total_token_count)
        return [], 0
    return items, total_token_count


def _flush_remaining_batch(
    *,
    items: list[Any],
    total_token_count: int,
    submit_batch,
) -> None:
    if items:
        submit_batch(items, total_token_count)


def _build_contextual_hints_block(
    title_paragraph: PdfParagraph | None,
    local_title_paragraph: PdfParagraph | None,
) -> str:
    contextual_lines: list[str] = []
    hint_idx = 1
    if title_paragraph:
        contextual_lines.append(
            f"{hint_idx}. First title in full text: {title_paragraph.unicode}"
        )
        hint_idx += 1

    if local_title_paragraph:
        is_different_from_global = True
        if title_paragraph and local_title_paragraph.debug_id == title_paragraph.debug_id:
            is_different_from_global = False

        if is_different_from_global:
            contextual_lines.append(
                f"{hint_idx}. The most recent title is: {local_title_paragraph.unicode}"
            )

    if not contextual_lines:
        return ""
    return "## Contextual Hints for Better Translation\n" + "\n".join(
        contextual_lines
    ) + "\n"


def _build_batch_glossary_block(
    cached_glossaries: list | None,
    text_for_glossary_matching: str,
) -> str:
    glossary_entries_per_glossary: dict[str, list[tuple[str, str]]] = {}
    if cached_glossaries:
        for glossary in cached_glossaries:
            active_entries = glossary.get_active_entries_for_text(text_for_glossary_matching)
            if active_entries:
                glossary_entries_per_glossary[glossary.name] = sorted(active_entries)

    if not glossary_entries_per_glossary:
        return ""

    glossary_block_lines: list[str] = [
        "## Glossary",
        "If a glossary is provided:",
        "- Always use the exact target term.",
        "- Apply glossary items even inside tags or when broken by hyphens/line breaks.",
        "- If glossary does NOT include a term, translate it naturally.",
        "",
        "## Glossary Tables",
        "",
    ]

    for glossary_name, entries in glossary_entries_per_glossary.items():
        glossary_block_lines.append(f"### Glossary: {glossary_name}")
        glossary_block_lines.append("")
        glossary_block_lines.append(
            "| Source Term | Target Term |\n|-------------|-------------|"
        )
        for original_source, target_text in entries:
            glossary_block_lines.append(f"| {original_source} | {target_text} |")
        glossary_block_lines.append("")

    return "\n".join(glossary_block_lines)


def _build_reference_glossary_section(user_glossaries: list | None, inputs: list[str]) -> str:
    if not user_glossaries:
        return ""

    text_for_glossary = "\n\n".join(inputs)
    glossary_entries: dict[str, Any] = {}
    for glossary in user_glossaries:
        active_entries = glossary.get_active_entries_for_text(text_for_glossary)
        if active_entries:
            glossary_entries[glossary.name] = active_entries

    if not glossary_entries:
        return ""

    reference_glossary_section = "Reference Glossaries (for consistency and quality):\n"
    for glossary_name, entries in glossary_entries.items():
        reference_glossary_section += f"\n{glossary_name}:\n"
        for src, tgt in sorted(set(entries)):
            reference_glossary_section += f"- {src} -> {tgt}\n"

    reference_glossary_section += (
        "\nPlease consider these existing translations for consistency when "
        "extracting new terms. IMPORTANT: You should also extract terms that "
        "appear in the reference glossaries above if they are found in the input "
        "text - don't skip them just because they already exist in the reference."
    )
    return reference_glossary_section


class PatchedILTranslator(BabelDOCILTranslator):
    def _build_role_block(self) -> str:
        prompt_bundle = _get_prompt_bundle(self.translation_config)
        return build_babeldoc_role_block(
            prompt_bundle=prompt_bundle,
            lang_out=self.translation_config.lang_out,
            custom_system_prompt=getattr(
                self.translation_config,
                "custom_system_prompt",
                None,
            ),
        )

    def generate_prompt_for_llm(
        self,
        text: str,
        title_paragraph: PdfParagraph | None = None,
        local_title_paragraph: PdfParagraph | None = None,
        translate_input=None,
    ) -> str:
        prompt_bundle = _get_prompt_bundle(self.translation_config)
        return prompt_bundle.render(
            "translation.paragraph_prompt",
            role_block=self._build_role_block(),
            glossary_block=self._build_glossary_block(text),
            context_block=self._build_context_block(
                title_paragraph,
                local_title_paragraph,
                translate_input,
            ),
            lang_out=self.translation_config.lang_out,
            text_to_translate=text,
        )


class PatchedILTranslatorLLMOnly(BabelDOCILTranslatorLLMOnly):
    def process_page(
        self,
        page: Page,
        executor,
        pbar=None,
        tracker=None,
        executor2=None,
        translated_ids=None,
    ):
        self.translation_config.raise_if_cancelled()
        paragraphs = []
        total_token_count = 0
        page_font_map, page_xobj_font_map = self._build_font_maps(page)
        batch_token_limit = _get_babeldoc_param(
            self.translation_config,
            "paragraph_batch_token_limit",
            200,
        )
        batch_size_limit = _get_babeldoc_param(
            self.translation_config,
            "paragraph_batch_size_limit",
            5,
        )

        def submit_batch(batch_paragraphs: list[PdfParagraph], token_count: int) -> None:
            self.mid += 1
            executor.submit(
                self.translate_paragraph,
                LLMBatchParagraph(batch_paragraphs, [page] * len(batch_paragraphs), tracker),
                pbar,
                page_font_map,
                page_xobj_font_map,
                self.translation_config.shared_context_cross_split_part.first_paragraph,
                self.translation_config.shared_context_cross_split_part.recent_title_paragraph,
                executor2,
                priority=1048576 - token_count,
                paragraph_token_count=token_count,
                mp_id=self.mid,
            )

        for paragraph in page.pdf_paragraph:
            if _should_skip_paragraph(
                paragraph,
                translated_ids=translated_ids,
                min_text_length=self.translation_config.min_text_length,
            ):
                _advance_progress(pbar)
                continue

            total_token_count += self.calc_token_count(paragraph.unicode)
            paragraphs.append(paragraph)
            translated_ids.add(id(paragraph))
            if paragraph.layout_label == "title":
                self.shared_context_cross_split_part.recent_title_paragraph = (
                    copy.deepcopy(paragraph)
                )

            paragraphs, total_token_count = _flush_batch_if_needed(
                items=paragraphs,
                total_token_count=total_token_count,
                token_limit=batch_token_limit,
                size_limit=batch_size_limit,
                submit_batch=submit_batch,
            )

        _flush_remaining_batch(
            items=paragraphs,
            total_token_count=total_token_count,
            submit_batch=submit_batch,
        )

    def translate_paragraph(
        self,
        batch_paragraph,
        pbar=None,
        page_font_map: dict[str, PdfFont] = None,
        xobj_font_map: dict[int, dict[str, PdfFont]] = None,
        title_paragraph: PdfParagraph | None = None,
        local_title_paragraph: PdfParagraph | None = None,
        executor=None,
        paragraph_token_count: int = 0,
        mp_id: int = 0,
    ):
        self.translation_config.raise_if_cancelled()
        should_translate_paragraph = []
        llm_translate_trackers = []
        inputs = []
        same_as_input_min_input_tokens = _get_babeldoc_param(
            self.translation_config,
            "same_as_input_min_input_tokens",
            10,
        )
        output_ratio_min = _get_babeldoc_param(
            self.translation_config,
            "llm_output_ratio_min",
            0.3,
        )
        output_ratio_max = _get_babeldoc_param(
            self.translation_config,
            "llm_output_ratio_max",
            3.0,
        )
        edit_distance_threshold = _get_babeldoc_param(
            self.translation_config,
            "same_text_edit_distance_threshold",
            5,
        )
        same_text_min_input_tokens = _get_babeldoc_param(
            self.translation_config,
            "same_text_min_input_tokens",
            20,
        )
        try:
            paragraph_unicodes = []
            for i in range(len(batch_paragraph.paragraphs)):
                paragraph = batch_paragraph.paragraphs[i]
                tracker = batch_paragraph.trackers[i]
                text, translate_input = self.il_translator.pre_translate_paragraph(
                    paragraph,
                    tracker,
                    page_font_map,
                    xobj_font_map,
                )
                if text is None:
                    if pbar:
                        pbar.advance(1)
                    continue

                tracker.record_multi_paragraph_id(mp_id)
                llm_translate_tracker = tracker.new_llm_translate_tracker()
                should_translate_paragraph.append(i)
                llm_translate_trackers.append(llm_translate_tracker)
                inputs.append(
                    (
                        text,
                        translate_input,
                        paragraph,
                        tracker,
                        llm_translate_tracker,
                        paragraph_unicodes,
                    )
                )
                paragraph_unicodes.append(paragraph.unicode)

            if not inputs:
                return

            json_format_input = []
            for id_, input_text in enumerate(inputs):
                translate_input = input_text[1]
                tracker = input_text[3]
                tracker.record_multi_paragraph_index(id_)
                placeholders_hint = translate_input.get_placeholders_hint()
                obj = {
                    "id": id_,
                    "input": input_text[0],
                    "layout_label": input_text[2].layout_label,
                }
                if (
                    placeholders_hint
                    and self.translation_config.add_formula_placehold_hint
                ):
                    obj["formula_placeholders_hint"] = placeholders_hint
                json_format_input.append(obj)

            json_format_input_str = json.dumps(
                json_format_input,
                ensure_ascii=False,
                indent=2,
            )
            batch_text_for_glossary_matching = "\n".join(
                item.get("input", "") for item in json_format_input
            )
            final_input = self._build_llm_prompt(
                json_input_str=json_format_input_str,
                title_paragraph=title_paragraph,
                local_title_paragraph=local_title_paragraph,
                batch_text_for_glossary_matching=batch_text_for_glossary_matching,
            )

            for llm_translate_tracker in llm_translate_trackers:
                llm_translate_tracker.set_input(final_input)

            llm_output = self.translate_engine.llm_translate(
                final_input,
                rate_limit_params={
                    "paragraph_token_count": paragraph_token_count,
                    "request_json_mode": True,
                },
            )
            for llm_translate_tracker in llm_translate_trackers:
                llm_translate_tracker.set_output(llm_output)

            llm_output = self._clean_json_output(llm_output.strip())
            parsed_output = json.loads(llm_output)
            if isinstance(parsed_output, dict) and parsed_output.get(
                "output",
                parsed_output.get("input", False),
            ):
                parsed_output = [parsed_output]

            translation_results = {
                item["id"]: item.get("output", item.get("input"))
                for item in parsed_output
            }
            if len(translation_results) != len(inputs):
                raise Exception(
                    "Translation results length mismatch. "
                    f"Expected: {len(inputs)}, Got: {len(translation_results)}"
                )

            for id_, output in translation_results.items():
                should_fallback = True
                try:
                    if not isinstance(output, str):
                        logger.warning(
                            "Translation result is not a string. Output: %s",
                            output,
                        )
                        continue

                    id_ = int(id_)
                    if id_ >= len(inputs):
                        logger.warning("Invalid id %s, skipping", id_)
                        continue

                    translated_text = re.sub(r"[. 。…，]{20,}", ".", output)
                    translate_input = inputs[id_][1]
                    llm_translate_tracker = inputs[id_][4]
                    input_unicode = inputs[id_][0]
                    output_unicode = translated_text
                    trimed_input = re.sub(r"[. 。…，]{20,}", ".", input_unicode)
                    input_token_count = self.calc_token_count(trimed_input)
                    output_token_count = self.calc_token_count(output_unicode)

                    same_as_input = trimed_input == output_unicode
                    if (
                        same_as_input
                        and input_token_count > same_as_input_min_input_tokens
                        and not self.translation_config.disable_same_text_fallback
                    ):
                        llm_translate_tracker.set_error_message(
                            "Translation result is the same as input, fallback."
                        )
                        llm_translate_tracker.set_placeholder_full_match()
                        logger.warning(
                            "Translation result is the same as input, fallback."
                        )
                        continue

                    if not (
                        output_ratio_min
                        < output_token_count / input_token_count
                        < output_ratio_max
                    ):
                        llm_translate_tracker.set_error_message(
                            "Translation result is too long or too short. "
                            f"Input: {input_token_count}, Output: {output_token_count}"
                        )
                        logger.warning(
                            "Translation result is too long or too short. "
                            f"Input: {input_token_count}, Output: {output_token_count}"
                        )
                        llm_translate_tracker.set_placeholder_full_match()
                        continue

                    if not self.translation_config.disable_same_text_fallback:
                        edit_distance = Levenshtein.distance(
                            input_unicode,
                            output_unicode,
                        )
                        if (
                            edit_distance < edit_distance_threshold
                            and input_token_count > same_text_min_input_tokens
                        ):
                            llm_translate_tracker.set_error_message(
                                "Translation result edit distance is too small. "
                                f"distance: {edit_distance}, input: {input_unicode}, "
                                f"output: {output_unicode}"
                            )
                            logger.warning(
                                "Translation result edit distance is too small. "
                                f"distance: {edit_distance}, input: {input_unicode}, "
                                f"output: {output_unicode}"
                            )
                            llm_translate_tracker.set_placeholder_full_match()
                            continue

                    self.il_translator.post_translate_paragraph(
                        inputs[id_][2],
                        inputs[id_][3],
                        translate_input,
                        translated_text,
                    )
                    should_fallback = False
                    if pbar:
                        pbar.advance(1)
                except Exception as exc:
                    error_message = (
                        "Error translating paragraph. "
                        f"Error: {exc}."
                    )
                    logger.exception(error_message)
                    for llm_translate_tracker in llm_translate_trackers:
                        llm_translate_tracker.set_error_message(error_message)
                    continue
                finally:
                    self.total_count += 1
                    if should_fallback:
                        self.fallback_count += 1
                        inputs[id_][4].set_fallback_to_translate()
                        logger.warning(
                            "Fallback to simple translation. paragraph id: %s",
                            inputs[id_][2].debug_id,
                        )
                        paragraph_token_count = self.calc_token_count(
                            inputs[id_][2].unicode
                        )
                        paragraph_unicodes = inputs[id_][5]
                        inputs[id_][2].unicode = paragraph_unicodes[id_]
                        executor.submit(
                            self.il_translator.translate_paragraph,
                            inputs[id_][2],
                            batch_paragraph.pages[id_],
                            pbar,
                            inputs[id_][3],
                            page_font_map,
                            xobj_font_map,
                            priority=1048576 - paragraph_token_count,
                            paragraph_token_count=paragraph_token_count,
                            title_paragraph=title_paragraph,
                            local_title_paragraph=local_title_paragraph,
                        )
                    else:
                        self.ok_count += 1

        except Exception as exc:
            error_message = f"Error {exc} during translation. try fallback"
            logger.warning(error_message)
            for llm_translate_tracker in llm_translate_trackers:
                llm_translate_tracker.set_error_message(error_message)
                llm_translate_tracker.set_fallback_to_translate()
            self.total_count += len(llm_translate_trackers)
            self.fallback_count += len(llm_translate_trackers)
            for input_ in inputs:
                input_[2].unicode = input_[5]
            if not should_translate_paragraph:
                should_translate_paragraph = list(range(len(batch_paragraph.paragraphs)))
            for i in should_translate_paragraph:
                paragraph = batch_paragraph.paragraphs[i]
                tracker = batch_paragraph.trackers[i]
                if paragraph.debug_id is None:
                    continue
                paragraph_token_count = self.calc_token_count(paragraph.unicode)
                executor.submit(
                    self.il_translator.translate_paragraph,
                    paragraph,
                    batch_paragraph.pages[i],
                    pbar,
                    tracker,
                    page_font_map,
                    xobj_font_map,
                    priority=1048576 - paragraph_token_count,
                    paragraph_token_count=paragraph_token_count,
                    title_paragraph=title_paragraph,
                    local_title_paragraph=local_title_paragraph,
                )

    def _build_llm_prompt(
        self,
        json_input_str: str,
        title_paragraph: PdfParagraph | None,
        local_title_paragraph: PdfParagraph | None,
        batch_text_for_glossary_matching: str,
    ) -> str:
        prompt_bundle = _get_prompt_bundle(self.translation_config)
        glossary_block = _build_batch_glossary_block(
            self._cached_glossaries,
            batch_text_for_glossary_matching,
        )
        context_block = _build_contextual_hints_block(
            title_paragraph,
            local_title_paragraph,
        )
        return prompt_bundle.render(
            "translation.batch_prompt",
            role_block=build_babeldoc_role_block(
                prompt_bundle=prompt_bundle,
                lang_out=self.translation_config.lang_out,
                custom_system_prompt=getattr(
                    self.translation_config,
                    "custom_system_prompt",
                    None,
                ),
            ),
            glossary_block=glossary_block,
            context_block=context_block,
            lang_out=self.translation_config.lang_out,
            json_input_str=json_input_str,
            batch_text_for_glossary_matching=batch_text_for_glossary_matching,
        )


class PatchedAutomaticTermExtractor(BabelDOCAutomaticTermExtractor):
    def process_page(
        self,
        page: Page,
        executor,
        pbar=None,
        tracker: PageTermExtractTracker = None,
    ):
        self.translation_config.raise_if_cancelled()
        paragraphs = []
        total_token_count = 0
        batch_token_limit = _get_babeldoc_param(
            self.translation_config,
            "term_batch_token_limit",
            600,
        )
        batch_size_limit = _get_babeldoc_param(
            self.translation_config,
            "term_batch_size_limit",
            12,
        )

        def submit_batch(batch_paragraphs: list[PdfParagraph], token_count: int) -> None:
            executor.submit(
                self.extract_terms_from_paragraphs,
                TermBatchParagraph(batch_paragraphs, tracker),
                pbar,
                token_count,
                priority=1048576 - token_count,
            )

        for paragraph in page.pdf_paragraph:
            if _should_skip_paragraph(paragraph):
                _advance_progress(pbar)
                continue

            total_token_count += self.calc_token_count(paragraph.unicode)
            paragraphs.append(paragraph)
            paragraphs, total_token_count = _flush_batch_if_needed(
                items=paragraphs,
                total_token_count=total_token_count,
                token_limit=batch_token_limit,
                size_limit=batch_size_limit,
                submit_batch=submit_batch,
            )

        _flush_remaining_batch(
            items=paragraphs,
            total_token_count=total_token_count,
            submit_batch=submit_batch,
        )

    def extract_terms_from_paragraphs(
        self,
        paragraphs,
        pbar=None,
        paragraph_token_count: int = 0,
    ):
        self.translation_config.raise_if_cancelled()
        try:
            inputs = [p.unicode for p in paragraphs.paragraphs if p.unicode]
            tracker = paragraphs.tracker
            for unicode_text in inputs:
                tracker.append_paragraph_unicode(unicode_text)
            if not inputs:
                return

            prompt_bundle = _get_prompt_bundle(self.translation_config)
            prompt = prompt_bundle.render(
                "term_extraction.extract_terms_prompt",
                target_language=self.translation_config.lang_out,
                text_to_process="\n\n".join(inputs),
                reference_glossary_section=_build_reference_glossary_section(
                    self.shared_context.user_glossaries,
                    inputs,
                ),
                example_output="""[
  {"src": "LLM", "tgt": "大语言模型"},
  {"src": "GPT", "tgt": "GPT"}
]""",
            )
            tracker.set_input(prompt)
            output = self.translate_engine.llm_translate(
                prompt,
                rate_limit_params={
                    "paragraph_token_count": paragraph_token_count,
                    "request_json_mode": True,
                },
            )
            tracker.set_output(output)
            cleaned_output = self._clean_json_output(output)
            response = json.loads(cleaned_output)
            if not isinstance(response, list):
                response = [response]

            for term in response:
                if isinstance(term, dict) and "src" in term and "tgt" in term:
                    src_term = str(term["src"]).strip()
                    tgt_term = str(term["tgt"]).strip()
                    if src_term == tgt_term and len(src_term) < 3:
                        continue
                    if src_term and tgt_term and len(src_term) < 100:
                        self.shared_context.add_raw_extracted_term_pair(
                            src_term,
                            tgt_term,
                        )
        except Exception as exc:
            logger.warning("Error during automatic terms extract: %s", exc)
            return
        finally:
            if pbar:
                pbar.advance(len(paragraphs.paragraphs))


def apply_babeldoc_runtime_patch() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    validate_babeldoc_patch_targets()

    il_translator_module.ILTranslator = PatchedILTranslator
    il_translator_llm_only_module.ILTranslator = PatchedILTranslator
    il_translator_llm_only_module.ILTranslatorLLMOnly = PatchedILTranslatorLLMOnly
    automatic_term_extractor_module.AutomaticTermExtractor = (
        PatchedAutomaticTermExtractor
    )

    babeldoc_high_level.ILTranslator = PatchedILTranslator
    babeldoc_high_level.ILTranslatorLLMOnly = PatchedILTranslatorLLMOnly
    babeldoc_high_level.AutomaticTermExtractor = PatchedAutomaticTermExtractor
    _PATCH_APPLIED = True
