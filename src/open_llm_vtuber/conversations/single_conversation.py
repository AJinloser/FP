from typing import Union, List, Dict, Any, Optional, Tuple
import asyncio
import json
from loguru import logger
import numpy as np

from .conversation_utils import (
    create_batch_input,
    process_agent_output,
    send_conversation_start_signals,
    process_user_input,
    finalize_conversation_turn,
    cleanup_conversation,
    EMOJI_LIST,
)
from .types import WebSocketSend
from .tts_manager import TTSTaskManager
from ..chat_history_manager import store_message
from ..service_context import ServiceContext


async def process_single_conversation(
    context: ServiceContext,
    websocket_send: WebSocketSend,
    client_uid: str,
    user_input: Union[str, np.ndarray],
    selection: str = None,
    images: Optional[List[Dict[str, Any]]] = None,
    session_emoji: str = np.random.choice(EMOJI_LIST),
) -> str:
    """Process a single-user conversation turn

    Args:
        context: Service context containing all configurations and engines
        websocket_send: WebSocket send function
        client_uid: Client unique identifier
        user_input: Text or audio input from user
        selection: Selection for the conversation
        images: Optional list of image data
        session_emoji: Emoji identifier for the conversation

    Returns:
        str: Complete response text
    """
    # Create TTSTaskManager for this conversation
    tts_manager = TTSTaskManager()

    try:
        # Send initial signals
        if not isinstance(user_input, np.ndarray):
            await send_conversation_start_signals(websocket_send)
        logger.info(f"New Conversation Chain {session_emoji} started!")

        # Process user input
        input_text = await process_user_input(
            user_input, context.asr_engine, websocket_send
        )
        
        # 如果是语音输入，直接返回，不继续处理
        if isinstance(user_input, np.ndarray):
            return ""

        # 只有文本输入才继续处理
        if input_text:
            # Create batch input
            batch_input = create_batch_input(
                input_text=input_text,
                images=images,
                from_name=context.character_config.human_name,
                selection=selection
            )

            # Store user message - 使用 user_id 而不是 conf_uid
            if context.history_uid:
                store_message(
                    user_id=context.agent_engine.get_conversation_info()["user_id"],
                    history_uid=context.history_uid,
                    role="human",
                    content=input_text,
                    name=context.character_config.human_name,
                )
            logger.info(f"User input: {input_text}")
            if images:
                logger.info(f"With {len(images)} images")

            # Process agent response
            full_response, message_id = await process_agent_response(
                context=context,
                batch_input=batch_input,
                websocket_send=websocket_send,
                tts_manager=tts_manager,
            )

            # Wait for any pending TTS tasks
            if tts_manager.task_list:
                await asyncio.gather(*tts_manager.task_list)
                await websocket_send(json.dumps({"type": "backend-synth-complete"}))

            await finalize_conversation_turn(
                tts_manager=tts_manager,
                websocket_send=websocket_send,
                client_uid=client_uid,
            )

            # Store AI response - 使用 user_id 而不是 conf_uid
            if context.history_uid and full_response:
                store_message(
                    user_id=context.agent_engine.get_conversation_info()["user_id"],
                    history_uid=context.history_uid,
                    role="ai",
                    content=full_response,
                    name=context.character_config.character_name,
                    avatar=context.character_config.avatar,
                    message_id=message_id
                )
                logger.info(f"AI response stored with message_id: {message_id}")
            
            logger.info(f"AI response: {full_response}")

            return full_response

    except asyncio.CancelledError:
        logger.info(f"🤡👍 Conversation {session_emoji} cancelled because interrupted.")
        raise
    except Exception as e:
        logger.error(f"Error in conversation chain: {e}")
        await websocket_send(
            json.dumps({"type": "error", "message": f"Conversation error: {str(e)}"})
        )
        raise
    finally:
        cleanup_conversation(tts_manager, session_emoji)


async def process_agent_response(
    context: ServiceContext,
    batch_input: Any,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
) -> Tuple[str, Optional[str]]:
    """Process agent response and generate output

    Args:
        context: Service context containing all configurations and engines
        batch_input: Input data for the agent
        websocket_send: WebSocket send function
        tts_manager: TTSTaskManager for the conversation

    Returns:
        Tuple[str, Optional[str]]: The complete response text and the message ID
    """
    full_response = ""
    message_id = None
    try:
        agent_output = context.agent_engine.chat(batch_input)
        async for output in agent_output:
            response_part, current_message_id = await process_agent_output(
                output=output,
                character_config=context.character_config,
                live2d_model=context.live2d_model,
                tts_engine=context.tts_engine,
                websocket_send=websocket_send,
                tts_manager=tts_manager,
                translate_engine=context.translate_engine,
            )
            full_response += response_part
            if current_message_id:
                message_id = current_message_id

    except Exception as e:
        logger.error(f"Error processing agent response: {e}")
        raise

    return full_response, message_id
