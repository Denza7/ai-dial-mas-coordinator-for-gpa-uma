import json
from typing import Optional

import httpx
from aidial_sdk.chat_completion import Role, Request, Message, Stage, Choice
from pydantic import StrictStr


_UMS_CONVERSATION_ID = "ums_conversation_id"


class UMSAgentGateway:

    def __init__(self, ums_agent_endpoint: str):
        self.ums_agent_endpoint = ums_agent_endpoint

    async def response(
            self,
            choice: Choice,
            stage: Stage,
            request: Request,
            additional_instructions: Optional[str] = None
    ) -> Message:
        # Инициализируем или восстанавливаем идентификатор диалога UMS
        session_id = self.__get_ums_conversation_id(request)

        if session_id is None:
            session_id = await self.__create_ums_conversation()
            stage.append_content(f"> **System:** Established new UMS session context: `{session_id}`\n\n")

        # Синтезируем входящую нагрузку для агента
        raw_prompt = request.messages[-1].content
        enriched_prompt = (
            f"{raw_prompt}\n\n[System Directive]: {additional_instructions}"
            if additional_instructions else raw_prompt
        )

        agent_output = await self.__call_ums_agent(
            conversation_id=session_id,
            user_message=enriched_prompt,
            stage=stage
        )

        choice.set_state({_UMS_CONVERSATION_ID: session_id})

        return Message(
            role=Role.ASSISTANT,
            content=StrictStr(agent_output),
        )

    def __get_ums_conversation_id(self, request: Request) -> Optional[str]:
        return next(
            (
                msg.custom_content.state.get(_UMS_CONVERSATION_ID)
                for msg in request.messages
                if msg.custom_content and msg.custom_content.state
            ),
            None
        )

    async def __create_ums_conversation(self) -> str:
        target_url = f"{self.ums_agent_endpoint.rstrip('/')}/conversations"
        payload = {"title": "Managed User Session"}

        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as session:

            response = await session.post(target_url, json=payload)
            response.raise_for_status()

            record = response.json()
            return str(record.get('id', ''))

    async def __call_ums_agent(
            self,
            conversation_id: str,
            user_message: str,
            stage: Stage
    ) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.ums_agent_endpoint}/conversations/{conversation_id}/chat",
                json={
                    "message": {
                        "role": "user",
                        "content": user_message
                    },
                    "stream": True
                },
                timeout=60.0
            )
            response.raise_for_status()

            content = ''
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    data_str = line[6:]

                    if data_str == '[DONE]':
                        break

                    try:
                        data = json.loads(data_str)

                        if 'conversation_id' in data:
                            continue

                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            if delta_content := delta.get('content'):
                                stage.append_content(delta_content)
                                content += delta_content
                    except json.JSONDecodeError:
                        continue

            return content