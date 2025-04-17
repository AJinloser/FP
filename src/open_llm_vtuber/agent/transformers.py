from typing import AsyncIterator, Tuple, Callable, List
from functools import wraps
from .output_types import Actions, SentenceOutput, DisplayText
from ..utils.tts_preprocessor import tts_filter as filter_text
from ..live2d_model import Live2dModel
from ..config_manager import TTSPreprocessorConfig
from ..utils.sentence_divider import SentenceDivider
from ..utils.sentence_divider import SentenceWithTags, TagState
from loguru import logger


def sentence_divider(
    faster_first_response: bool = True,
    segment_method: str = "pysbd",
    valid_tags: List[str] = None,
):
    """
    Decorator that transforms token stream into sentences with tags
    """
    def decorator(
        func: Callable[..., AsyncIterator[str]],
    ) -> Callable[..., AsyncIterator[SentenceWithTags]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> AsyncIterator[SentenceWithTags]:
            # 直接将输入流转换为 SentenceWithTags，不进行分割
            token_stream = func(*args, **kwargs)
            async for token in token_stream:
                # 每个token作为一个完整的句子传递，保持原始格式
                yield SentenceWithTags(
                    text=token,
                    tags=[],  # 空标签列表，因为我们不需要处理标签
                )
                logger.debug(f"sentence_divider: {token}")

        return wrapper
    return decorator


def actions_extractor(live2d_model: Live2dModel):
    """
    Decorator that extracts actions from sentences
    """

    def decorator(
        func: Callable[..., AsyncIterator[SentenceWithTags]],
    ) -> Callable[..., AsyncIterator[Tuple[SentenceWithTags, Actions]]]:
        @wraps(func)
        async def wrapper(
            *args, **kwargs
        ) -> AsyncIterator[Tuple[SentenceWithTags, Actions]]:
            sentence_stream = func(*args, **kwargs)
            async for sentence in sentence_stream:
                actions = Actions()
                # Only extract emotions for non-tag text
                if not any(
                    tag.state in [TagState.START, TagState.END] for tag in sentence.tags
                ):
                    expressions = live2d_model.extract_emotion(sentence.text)
                    if expressions:
                        actions.expressions = expressions
                yield sentence, actions

        return wrapper

    return decorator


def display_processor():
    """
    Decorator that processes text for display.
    """
    def decorator(
        func: Callable[..., AsyncIterator[Tuple[SentenceWithTags, Actions]]],
    ) -> Callable[..., AsyncIterator[Tuple[SentenceWithTags, DisplayText, Actions]]]:
        @wraps(func)
        async def wrapper(
            *args, **kwargs
        ) -> AsyncIterator[Tuple[SentenceWithTags, DisplayText, Actions]]:
            stream = func(*args, **kwargs)

            async for sentence, actions in stream:
                # 直接使用原始文本，不做任何修改
                display = DisplayText(text=sentence.text)
                yield sentence, display, actions

        return wrapper
    return decorator


def tts_filter(
    tts_preprocessor_config: TTSPreprocessorConfig = None,
):
    """
    Decorator that filters text for TTS.
    Skips TTS for think tag content.
    """

    def decorator(
        func: Callable[
            ..., AsyncIterator[Tuple[SentenceWithTags, DisplayText, Actions]]
        ],
    ) -> Callable[..., AsyncIterator[SentenceOutput]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> AsyncIterator[SentenceOutput]:
            sentence_stream = func(*args, **kwargs)
            config = tts_preprocessor_config or TTSPreprocessorConfig()

            async for sentence, display, actions in sentence_stream:
                if any(tag.name == "think" for tag in sentence.tags):
                    tts = ""
                else:
                    tts = filter_text(
                        text=display.text,
                        remove_special_char=config.remove_special_char,
                        ignore_brackets=config.ignore_brackets,
                        ignore_parentheses=config.ignore_parentheses,
                        ignore_asterisks=config.ignore_asterisks,
                        ignore_angle_brackets=config.ignore_angle_brackets,
                    )

                logger.debug(f"[{display.name}] display: {display.text}")
                logger.debug(f"[{display.name}] tts: {tts}")

                yield SentenceOutput(
                    display_text=display,
                    tts_text=tts,
                    actions=actions,
                )

        return wrapper

    return decorator
