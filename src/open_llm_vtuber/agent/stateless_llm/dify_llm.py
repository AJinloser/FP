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
        按顺序划分语句，处理普通文本、代码块和表格。
        返回一个列表：[完整的句子/代码块/表格, 剩余的不完整内容]
        """
        if not text.strip():
            return []

        # 查找第一个特殊标记（代码块或表格）的位置
        code_start = text.find('```')
        table_start = text.find('|')
        
        # 没有特殊标记，按普通文本处理
        if code_start == -1 and table_start == -1:
            return self._split_normal_text(text)
        
        # 确定第一个特殊标记的位置和类型
        first_special = min(
            (pos for pos in [code_start, table_start] if pos != -1), 
            default=-1
        )
        
        # 如果特殊标记前有普通文本，强制输出
        if first_special > 0:
            normal_text = text[:first_special]
            # 在普通文本末尾添加换行符
            if not normal_text.endswith('\n'):
                normal_text += '\n'
            return [normal_text, text[first_special:]]
        
        # 处理代码块
        if first_special == code_start:
            if text.count('```', code_start) < 2:
                # 代码块未完成
                return []
            
            end_idx = text.find('```', code_start + 3)
            code_block = text[code_start:end_idx + 3]
            remaining = text[end_idx + 3:]
            
            # 在代码块结束后添加换行符
            if remaining and not remaining.startswith('\n'):
                code_block += '\n'
            
            return [code_block, remaining]
        
        # 处理表格
        if first_special == table_start:
            lines = text[table_start:].split('\n')
            table_lines = []
            remaining_lines = []
            in_table = False
            
            for i, line in enumerate(lines):
                if line.strip().startswith('|'):
                    if not in_table:
                        in_table = True
                    table_lines.append(line)
                else:
                    if in_table:
                        # 表格结束
                        remaining_lines = lines[i:]
                        break
                    remaining_lines.append(line)
            
            if table_lines:
                table = '\n'.join(table_lines)
                remaining = '\n'.join(remaining_lines)
                return [table, remaining]
            
        return []

    def _split_normal_text(self, text: str) -> List[str]:
        """
        处理普通文本的分句，只使用中文标点作为分割符
        """
        # 只使用中文句号、问号和感叹号作为分割符
        sentence_endings = ['。', '？', '！']
        
        # 从后向前查找句子结束符
        for i in range(len(text)-1, -1, -1):
            if text[i] in sentence_endings:
                # 确保这是句子的真正结束
                if i + 1 == len(text) or text[i + 1] in ['\n', ' ', '']:
                    return [text[:i+1], text[i+1:]]
        
        # 如果没有找到完整句子，返回空列表
        return [] 