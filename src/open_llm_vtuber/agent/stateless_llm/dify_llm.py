from typing import AsyncIterator, List, Dict, Any
import json
import aiohttp
from loguru import logger

from .stateless_llm_interface import StatelessLLMInterface

class AsyncLLM(StatelessLLMInterface):
    def __init__(
        self,
        base_url: str,
        llm_api_key: str,
        model: str = "default",  # Dify中model参数不需要显式指定
        temperature: float = 1.0,
    ):
        """初始化 Dify LLM 实例

        Args:
            base_url (str): Dify API 的基础 URL
            llm_api_key (str): API 密钥
            model (str, optional): 模型名称（在Dify中通常不需要）
            temperature (float, optional): 采样温度
        """
        self.base_url = base_url.rstrip("/")
        self.chat_endpoint = f"{self.base_url}/v1/chat-messages"
        self.headers = {
            "Authorization": f"Bearer {llm_api_key}",
            "Content-Type": "application/json",
        }
        self.model = model
        self.temperature = temperature
        
        logger.info(f"已初始化 Dify LLM，API端点：{self.chat_endpoint}")

    async def chat_completion(
        self, messages: List[Dict[str, Any]], system: str = None
    ) -> AsyncIterator[str]:
        """生成聊天回复

        Args:
            messages (List[Dict[str, Any]]): 消息列表
            system (str, optional): 系统提示词

        Yields:
            str: API 响应的每个文本块
        """
        try:
            # 构建最后一条用户消息
            last_message = messages[-1]["content"] if messages else ""
            
            # 准备请求数据
            data = {
                "inputs": {},  # Dify的inputs参数，默认为空
                "query": last_message,
                "response_mode": "streaming",
                "user": "default_user",  # 可以根据需要设置用户标识
            }

            # 如果有system prompt，添加到inputs中
            if system:
                data["inputs"]["system"] = system

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.chat_endpoint,
                    headers=self.headers,
                    json=data
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Dify API返回错误: {response.status} - {error_text}")

                    # 处理SSE流式响应
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if not line or line == "":
                            continue
                        if line.startswith("data: "):
                            try:
                                event_data = json.loads(line[6:])  # 去掉"data: "前缀
                                
                                if event_data.get("event") == "error":
                                    error_msg = event_data.get("message", "未知错误")
                                    logger.error(f"Dify API错误: {error_msg}")
                                    yield f"错误: {error_msg}"
                                    break
                                
                                elif event_data.get("event") == "message":
                                    answer = event_data.get("answer", "")
                                    if answer:
                                        yield answer
                                
                                elif event_data.get("event") == "message_end":
                                    break

                            except json.JSONDecodeError as e:
                                logger.error(f"解析响应数据时出错: {e}")
                                continue

        except Exception as e:
            logger.error(f"调用 Dify API 时发生错误: {e}")
            yield f"调用 Dify API 时发生错误: {e}" 