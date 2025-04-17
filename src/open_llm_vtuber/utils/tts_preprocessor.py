import re
import unicodedata
from loguru import logger
from ..translate.translate_interface import TranslateInterface


def tts_filter(
    text: str,
    remove_special_char: bool,
    ignore_brackets: bool,
    ignore_parentheses: bool,
    ignore_asterisks: bool,
    ignore_angle_brackets: bool,
    translator: TranslateInterface | None = None,
) -> str:
    """
    Filter or do anything to the text before TTS generates the audio.
    Changes here do not affect subtitles or LLM's memory. The generated audio is
    the only affected thing.

    Args:
        text (str): The text to filter.
        remove_special_char (bool): Whether to remove special characters.
        ignore_brackets (bool): Whether to ignore text within brackets.
        ignore_parentheses (bool): Whether to ignore text within parentheses.
        ignore_asterisks (bool): Whether to ignore text within asterisks.
        translator (TranslateInterface, optional):
            The translator to use. If None, we'll skip the translation. Defaults to None.

    Returns:
        str: The filtered text.
    """
    if ignore_asterisks:
        try:
            text = filter_asterisks(text)
        except Exception as e:
            logger.warning(f"Error ignoring asterisks: {e}")
            logger.warning(f"Text: {text}")
            logger.warning("Skipping...")

    if ignore_brackets:
        try:
            text = filter_brackets(text)
        except Exception as e:
            logger.warning(f"Error ignoring brackets: {e}")
            logger.warning(f"Text: {text}")
            logger.warning("Skipping...")
    if ignore_parentheses:
        try:
            text = filter_parentheses(text)
        except Exception as e:
            logger.warning(f"Error ignoring parentheses: {e}")
            logger.warning(f"Text: {text}")
            logger.warning("Skipping...")
    if ignore_angle_brackets:
        try:
            text = filter_angle_brackets(text)
        except Exception as e:
            logger.warning(f"Error ignoring angle brackets: {e}")
            logger.warning(f"Text: {text}")
            logger.warning("Skipping...")
    if remove_special_char:
        try:
            text = remove_special_characters(text)
        except Exception as e:
            logger.warning(f"Error removing special characters: {e}")
            logger.warning(f"Text: {text}")
            logger.warning("Skipping...")
    if translator:
        try:
            logger.info("Translating...")
            text = translator.translate(text)
            logger.info(f"Translated: {text}")
        except Exception as e:
            logger.critical(f"Error translating: {e}")
            logger.critical(f"Text: {text}")
            logger.warning("Skipping...")

    logger.debug(f"Filtered text: {text}")
    return text


def remove_special_characters(text: str) -> str:
    """
    Filter text to keep Chinese characters, English letters, numbers, spaces,
    and basic mathematical symbols, while handling markdown and LaTeX formatting.
    Skip table content to prevent TTS errors.
    """
    # 首先处理表格分隔符中的连续减号
    text = re.sub(r'\|[\s-]*\|', '||', text)  # 替换表格分隔行
    
    # 其他文本处理保持不变
    text = re.sub(r'\\text{([^}]*)}', r'\1', text)
    text = re.sub(r'\\\[|\\\]', '', text)  # 移除 \[ \] 数学环境标记
    text = re.sub(r'\$\$.*?\$\$', '', text)  # 移除 $$ 数学环境
    text = re.sub(r'\$.*?\$', '', text)  # 移除 $ 数学环境
    text = re.sub(r'\\[a-zA-Z]+(?:\{[^}]*\})?', '', text)  # 移除其他 LaTeX 命令

    # 处理 markdown 格式
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # 移除加粗标记但保留内容
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # 移除斜体标记但保留内容
    text = re.sub(r'`([^`]+)`', r'\1', text)        # 移除代码标记但保留内容
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)  # 移除标题标记
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)   # 处理链接，保留显示文本

    normalized_text = unicodedata.normalize("NFKC", text)

    def is_valid_char(char: str) -> bool:
        category = unicodedata.category(char)
        
        # 定义允许的数学和特殊符号
        allowed_symbols = {
            '+',   # 加号
            '-',   # 减号（数学运算符）
            '×',   # 乘号
            '÷',   # 除号
            '/',   # 斜杠（除号）
            '=',   # 等号
            '%',   # 百分号
            '.',   # 小数点
            '。',  # 句号（中文）
            '，',  # 逗号（中文）
            ',',   # 逗号
            '：',  # 冒号（中文）
            ':',   # 冒号
            '！',  # 感叹号（中文）
            '!',   # 感叹号
            '？',  # 问号（中文）
            '?',   # 问号
        }

        # 如果是连续的减号，则不保留
        if char == '-' and normalized_text.find('---') != -1:
            return False

        return (
            category in {'Lo', 'Ll', 'Lu', 'Nd'}  # 允许中文字符、英文字母和数字
            or char.isspace()  # 保留空格
            or char in allowed_symbols  # 允许特定的数学符号和标点
        )

    filtered_text = "".join(char for char in normalized_text if is_valid_char(char))
    
    # 清理多余的空白字符
    filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()
    
    # 添加日志以便调试
    logger.debug(f"Original text: {text}")
    logger.debug(f"Filtered text for TTS: {filtered_text}")
    
    return filtered_text


def _filter_nested(text: str, left: str, right: str) -> str:
    """
    Generic function to handle nested symbols.

    Args:
        text (str): The text to filter.
        left (str): The left symbol (e.g. '[' or '(').
        right (str): The right symbol (e.g. ']' or ')').

    Returns:
        str: The filtered text.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    if not text:
        return text

    result = []
    depth = 0
    for char in text:
        if char == left:
            depth += 1
        elif char == right:
            if depth > 0:
                depth -= 1
        else:
            if depth == 0:
                result.append(char)
    filtered_text = "".join(result)
    filtered_text = re.sub(r"\s+", " ", filtered_text).strip()
    return filtered_text


def filter_brackets(text: str) -> str:
    """
    Filter text to remove all text within brackets, handling nested cases.

    Args:
        text (str): The text to filter.

    Returns:
        str: The filtered text.
    """
    return _filter_nested(text, "[", "]")


def filter_parentheses(text: str) -> str:
    """
    Filter text to remove all text within parentheses, handling nested cases.

    Args:
        text (str): The text to filter.

    Returns:
        str: The filtered text.
    """
    return _filter_nested(text, "(", ")")


def filter_angle_brackets(text: str) -> str:
    """
    Filter text to remove all text within angle brackets, handling nested cases.

    Args:
        text (str): The text to filter.

    Returns:
        str: The filtered text.
    """
    return _filter_nested(text, "<", ">")


def filter_asterisks(text: str) -> str:
    """
    Removes text enclosed within asterisks of any length (*, **, ***, etc.) from a string.

    Args:
        text: The input string.

    Returns:
        The string with asterisk-enclosed text removed.
    """
    # Handle asterisks of any length (*, **, ***, etc.)
    filtered_text = re.sub(r"\*{1,}((?!\*).)*?\*{1,}", "", text)

    # Clean up any extra spaces
    filtered_text = re.sub(r"\s+", " ", filtered_text).strip()

    return filtered_text
