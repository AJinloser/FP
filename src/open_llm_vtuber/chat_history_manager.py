import os
import re
import json
import uuid
from datetime import datetime
from typing import Literal, List, TypedDict, Optional, Dict
from loguru import logger


class HistoryMessage(TypedDict):
    role: Literal["human", "ai"]
    timestamp: str
    content: str
    # Optional display information for the message
    name: Optional[str]
    avatar: Optional[str]


class HistoryMetadata(TypedDict):
    role: Literal["metadata"]
    timestamp: str
    conversation_id: Optional[str]  # Dify 会话 ID
    user_id: Optional[str]  # 用户标识


def _is_safe_filename(filename: str) -> bool:
    """Validate filename for safety and allowed characters"""
    if not filename or len(filename) > 255:
        return False

    # Allow alphanumeric, hyphen, underscore, and common unicode characters
    # Block any filesystem special characters, control characters, and path separators
    pattern = re.compile(r"^[\w\-_\u0020-\u007E\u00A0-\uFFFF]+$")
    return bool(pattern.match(filename))


def _sanitize_path_component(component: str) -> str:
    """Sanitize and validate a path component"""
    # Remove any path components, get just the basename
    sanitized = os.path.basename(component.strip())

    if not _is_safe_filename(sanitized):
        raise ValueError(f"Invalid characters in path component: {component}")

    return sanitized


def _ensure_user_dir(user_id: str) -> str:
    """确保用户目录存在并返回路径"""
    if not user_id:
        raise ValueError("user_id cannot be empty")

    safe_user_id = _sanitize_path_component(user_id)
    base_dir = os.path.join("chat_history", "users", safe_user_id)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _get_safe_history_path(user_id: str, history_uid: str) -> str:
    """Get sanitized path for history file"""
    safe_user_id = _sanitize_path_component(user_id)
    safe_history_uid = _sanitize_path_component(history_uid)
    base_dir = os.path.join("chat_history", "users", safe_user_id)
    full_path = os.path.normpath(os.path.join(base_dir, f"{safe_history_uid}.json"))
    if not full_path.startswith(base_dir):
        raise ValueError("Invalid path: Path traversal detected")
    return full_path


def create_new_history(user_id: str) -> str:
    """创建新的历史记录文件"""
    if not user_id:
        logger.warning("No user_id provided")
        return ""

    history_uid = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4().hex}"
    user_dir = _ensure_user_dir(user_id)

    try:
        filepath = os.path.join(user_dir, f"{history_uid}.json")
        initial_data = [
            {
                "role": "metadata",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "conversation_id": None,
                "user_id": user_id
            }
        ]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to create new history file: {e}")
        return ""

    logger.debug(f"Created new history file with metadata: {filepath}")
    return history_uid


def store_message(
    user_id: str,
    history_uid: str,
    role: Literal["human", "ai"],
    content: str,
    name: str | None = None,
    avatar: str | None = None,
):
    """Store a message in a specific history file

    Args:
        user_id: User identifier
        history_uid: History unique identifier
        role: Message role ("human" or "ai")
        content: Message content
        name: Optional display name (default None)
        avatar: Optional avatar URL (default None)
    """
    if not user_id or not history_uid:
        if not user_id:
            logger.warning("Missing user_id")
        if not history_uid:
            logger.warning("Missing history_uid")
        return

    filepath = _get_safe_history_path(user_id, history_uid)
    logger.debug(f"Storing {role} message to {filepath}")

    history_data = []
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                history_data = json.load(f)
        except Exception:
            logger.error(f"Failed to load history file: {filepath}")
            pass

    now_str = datetime.now().isoformat(timespec="seconds")
    new_item = {
        "role": role,
        "timestamp": now_str,
        "content": content,
    }

    # Add optional display information if provided
    if name is not None:
        new_item["name"] = name
    if avatar is not None:
        new_item["avatar"] = avatar

    history_data.append(new_item)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)
    logger.debug(f"Successfully stored {role} message")


def get_metadata(user_id: str, history_uid: str) -> dict:
    """Get metadata from history file"""
    if not user_id or not history_uid:
        return {}

    filepath = _get_safe_history_path(user_id, history_uid)
    if not os.path.exists(filepath):
        return {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            history_data = json.load(f)

        if history_data and history_data[0]["role"] == "metadata":
            return history_data[0]
    except Exception as e:
        logger.error(f"Failed to get metadata: {e}")
    return {}


def update_metadate(user_id: str, history_uid: str, metadata: dict) -> bool:
    """Set metadata in history file

    Updates existing metadata with new fields, preserving existing ones.
    If no metadata exists, creates new metadata entry.
    """
    if not user_id or not history_uid:
        return False

    filepath = _get_safe_history_path(user_id, history_uid)
    if not os.path.exists(filepath):
        return False

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            history_data = json.load(f)

        if history_data and history_data[0]["role"] == "metadata":
            # 更新现有元数据，保留其他字段
            # if "conversation_id" in metadata:
            #     logger.info(f"更新对话历史元数据 - 新的 conversation_id: {metadata['conversation_id']}")
            history_data[0].update(metadata)
        else:
            # 如果没有元数据，创建新的
            new_metadata = {
                "role": "metadata",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            new_metadata.update(metadata)
            history_data.insert(0, new_metadata)
            logger.info(f"创建新的对话历史元数据: {new_metadata}")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)

        logger.debug(f"更新对话历史 {history_uid} 的元数据成功")
        return True
    except Exception as e:
        logger.error(f"更新元数据失败: {e}")
    return False


def get_history(user_id: str, history_uid: str) -> List[HistoryMessage]:
    """Read chat history for the given user_id and history_uid"""
    if not user_id or not history_uid:
        if not user_id:
            logger.warning("Missing user_id")
        if not history_uid:
            logger.warning("Missing history_uid")
        return []

    filepath = _get_safe_history_path(user_id, history_uid)

    if not os.path.exists(filepath):
        logger.warning(f"History file not found: {filepath}")
        return []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            history_data = json.load(f)
            # Filter out metadata
            return [msg for msg in history_data if msg["role"] != "metadata"]
    except Exception:
        return []


def delete_history(user_id: str, history_uid: str) -> bool:
    """Delete a specific history file"""
    if not user_id or not history_uid:
        logger.warning("Missing user_id or history_uid")
        return False

    filepath = _get_safe_history_path(user_id, history_uid)
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.debug(f"Successfully deleted history file: {filepath}")
            return True
    except Exception as e:
        logger.error(f"Failed to delete history file: {e}")
    return False


def get_history_list(user_id: str) -> List[dict]:
    """获取用户的历史记录列表"""
    if not user_id:
        return []

    histories = []
    user_dir = _ensure_user_dir(user_id)
    empty_history_uids = []

    try:
        for filename in os.listdir(user_dir):
            if not filename.endswith(".json"):
                continue

            history_uid = filename[:-5]
            filepath = os.path.join(user_dir, filename)

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    messages = json.load(f)

                    # Filter out metadata for checking if history is empty
                    actual_messages = [
                        msg for msg in messages if msg["role"] != "metadata"
                    ]
                    if not actual_messages:
                        empty_history_uids.append(history_uid)
                        continue

                    latest_message = actual_messages[-1]
                    history_info = {
                        "uid": history_uid,
                        "latest_message": latest_message,
                        "timestamp": (
                            latest_message["timestamp"] if latest_message else None
                        ),
                    }
                    histories.append(history_info)
            except Exception as e:
                logger.error(f"Error reading history file {filename}: {e}")
                continue

        # Clean up empty histories if there are other non-empty ones
        if len(empty_history_uids) > 0 and len(os.listdir(user_dir)) > 1:
            for uid in empty_history_uids:
                try:
                    os.remove(os.path.join(user_dir, f"{uid}.json"))
                    logger.info(f"Removed empty history file: {uid}")
                except Exception as e:
                    logger.error(f"Failed to remove empty history file {uid}: {e}")

        histories.sort(
            key=lambda x: x["timestamp"] if x["timestamp"] else "", reverse=True
        )
        return histories

    except Exception as e:
        logger.error(f"Error listing histories: {e}")
        return []


def modify_latest_message(
    user_id: str,
    history_uid: str,
    role: Literal["human", "ai", "system"],
    new_content: str,
) -> bool:
    """Modify the latest message in a specific history file if it matches the given role"""
    if not user_id or not history_uid:
        logger.warning("Missing user_id or history_uid")
        return False

    filepath = _get_safe_history_path(user_id, history_uid)
    if not os.path.exists(filepath):
        logger.warning(f"History file not found: {filepath}")
        return False

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            history_data = json.load(f)

        if not history_data:
            logger.warning("History is empty")
            return False

        latest_message = history_data[-1]
        if latest_message["role"] != role:
            logger.warning(
                f"Latest message role ({latest_message['role']}) doesn't match requested role ({role})"
            )
            return False

        latest_message["content"] = new_content
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)

        logger.debug(f"Successfully modified latest {role} message")
        return True

    except Exception as e:
        logger.error(f"Failed to modify latest message: {e}")
        return False


def rename_history_file(
    user_id: str, old_history_uid: str, new_history_uid: str
) -> bool:
    """Rename a history file with a new history_uid"""
    if not user_id or not old_history_uid or not new_history_uid:
        logger.warning("Missing required parameters for rename")
        return False

    old_filepath = _get_safe_history_path(user_id, old_history_uid)
    new_filepath = _get_safe_history_path(user_id, new_history_uid)

    try:
        if os.path.exists(old_filepath):
            os.rename(old_filepath, new_filepath)
            logger.info(
                f"Renamed history file from {old_history_uid} to {new_history_uid}"
            )
            return True
    except Exception as e:
        logger.error(f"Failed to rename history file: {e}")
    return False
