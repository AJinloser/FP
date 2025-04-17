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
        self.buffer = ""
        self.newline_buffer = ""  # 添加新的缓冲区用于处理换行符
        
        logger.info(f"已初始化 Dify LLM，API端点：{self.chat_endpoint}")

    async def chat_completion(
        self, 
        messages: List[Dict[str, Any]], 
        system: str = None,
        conversation_id: str = "",
        user_id: str = None
    ) -> AsyncIterator[str]:
        """生成聊天回复

        Args:
            messages (List[Dict[str, Any]]): 消息列表
            system (str, optional): 系统提示词
            conversation_id (str, optional): Dify 会话 ID
            user_id (str, optional): 用户标识

        Yields:
            str: API 响应的每个文本块
        """
        try:
            logger.info(f"准备发送请求到 Dify - conversation_id: {conversation_id}, user_id: {user_id}")
            
            # 构建最后一条用户消息
            last_message = messages[-1]["content"] if messages else ""
            
            # 准备请求数据
            data = {
                "inputs": {},
                "query": last_message,
                "response_mode": "streaming",
                "user": user_id,
                "conversation_id": conversation_id
            }

            # if system:
            #     data["inputs"]["system"] = system

            logger.info(f"Dify 请求数据: {data}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.chat_endpoint,
                    headers=self.headers,
                    json=data
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Dify API返回错误: {response.status} - {error_text}")
                        raise Exception(f"Dify API返回错误: {response.status} - {error_text}")

                    # 处理SSE流式响应
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if not line or line == "":
                            continue
                        if line.startswith("data: "):
                            try:
                                event_data = json.loads(line[6:])  # 去掉"data: "前缀
                                
                                # 处理conversation_id
                                if not conversation_id and event_data.get("conversation_id"):
                                    new_conversation_id = event_data["conversation_id"]
                                    yield f"__conversation_id:{new_conversation_id}"
                                
                                if event_data.get("event") == "error":
                                    error_msg = event_data.get("message", "未知错误")
                                    logger.error(f"Dify API错误: {error_msg}")
                                    yield f"错误: {error_msg}"
                                    break
                                
                                elif event_data.get("event") == "message":
                                    answer = event_data.get("answer", "")
                                    if answer:
                                        self.buffer += answer
                                        # 检查是否有完整的句子或段落
                                        sentences = self._split_complete_sentences(self.buffer)
                                        if sentences:
                                            complete_text = sentences[0]
                                            self.buffer = sentences[1]
                                            
                                            # 特殊处理代码块
                                            if '```' in complete_text:
                                                logger.info(f"Dify API 返回代码块: {complete_text}")
                                            else:
                                                logger.info(f"Dify API 返回文本: {complete_text}")
                                            yield complete_text
                                
                                elif event_data.get("event") == "message_end":
                                    # 输出剩余的缓冲区内容
                                    if self.buffer:
                                        logger.info(f"Dify API 返回最后内容: {self.buffer}")
                                        yield self.buffer
                                        self.buffer = ""
                                    break

                            except json.JSONDecodeError as e:
                                logger.error(f"解析响应数据时出错: {e}")
                                continue

        except Exception as e:
            logger.error(f"调用 Dify API 时发生错误: {e}")
            yield f"调用 Dify API 时发生错误: {e}"
        finally:
            # 清理缓冲区
            self.buffer = ""

    def _split_complete_sentences(self, text: str) -> List[str]:
        """
        将文本分割成完整的句子。
        返回一个列表：[完整的句子, 剩余的不完整内容]
        """
        # 检查是否在代码块内
        if '```' in text:
            # 如果发现代码块开始标记
            if text.count('```') == 1:
                # 只有开始标记，继续等待结束标记
                return []
            else:
                # 找到完整的代码块
                start_idx = text.find('```')
                end_idx = text.find('```', start_idx + 3)
                if end_idx != -1:
                    # 返回完整的代码块（包括结束标记）
                    return [text[:end_idx + 3], text[end_idx + 3:]]
                return []

        # 如果当前文本以"```"开头，等待更多内容
        if text.startswith('```'):
            return []

        # 检查是否在普通文本中
        # 检查常见的句子结束符
        sentence_endings = ['。', '！', '？', '…']  # 移除 '.', '!', '?' 避免代码中的符号被误判
        for i in range(len(text)-1, -1, -1):
            if text[i] in sentence_endings:
                # 确保这是真正的句子结束，而不是代码中的符号
                if i + 1 < len(text) and text[i + 1] in ['\n', ' ', '']:
                    return [text[:i+1], text[i+1:]]
            
        # 如果遇到换行符，且不是在代码块中
        if '\n' in text and not any(code_marker in text for code_marker in ['```python', '```']):
            lines = text.split('\n', 1)
            if lines[0].strip():  # 确保不是空行
                return [lines[0] + '\n', lines[1] if len(lines) > 1 else '']
        
        # # 如果文本长度超过阈值且包含完整句子的标志（如逗号），进行分割
        # if len(text) > 50 and ('，' in text or '；' in text):
        #     for i in range(len(text)-1, -1, -1):
        #         if text[i] in ['，', '；']:
        #             return [text[:i+1], text[i+1:]]
            
        # 如果没有找到完整句子，返回空列表
        return [] 