from copy import deepcopy
from typing import Optional, Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, CustomContent, Stage, Attachment
from pydantic import StrictStr

from task.stage_util import StageProcessor

_IS_GPA = "is_gpa"
_GPA_MESSAGES = "gpa_messages"


class GPAGateway:

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def response(
            self,
            choice: Choice,
            stage: Stage,
            request: Request,
            additional_instructions: Optional[str]
    ) -> Message:
        client_config = {
            "base_url": self.endpoint,
            "api_key": request.api_key,
            "api_version": '2025-01-01-preview'
        }
        dial_client = AsyncDial(**client_config)

        stream_response = await dial_client.chat.completions.create(
            stream=True,
            messages=self.__prepare_gpa_messages(request, additional_instructions),
            deployment_name="general-purpose-agent",
            extra_headers={'x-conversation-id': request.headers.get('x-conversation-id')}
        )

        full_text = []
        final_metadata = CustomContent(attachments=[])
        active_stages: dict[int, Stage] = {}

        async for chunk in stream_response:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if not delta:
                continue

            if delta.content:
                text_part = delta.content
                stage.append_content(text_part)
                full_text.append(text_part)

            if (extra_data := delta.custom_content):
                if extra_data.attachments:
                    final_metadata.attachments.extend(extra_data.attachments)
                if extra_data.state:
                    final_metadata.state = extra_data.state

                raw_payload = extra_data.model_dump(exclude_none=True)
                for stg_info in raw_payload.get("stages", []):
                    s_idx = stg_info["index"]

                    if s_idx not in active_stages:
                        active_stages[s_idx] = StageProcessor.open_stage(choice, stg_info.get("name"))

                    current_sub_stage = active_stages[s_idx]

                    if s_content := stg_info.get("content"):
                        current_sub_stage.append_content(s_content)
                    elif s_attachments := stg_info.get("attachments"):
                        for attachment_data in s_attachments:
                            current_sub_stage.add_attachment(Attachment(**attachment_data))
                    elif stg_info.get("status") == 'completed':
                        StageProcessor.close_stage_safely(current_sub_stage)

        for remaining_stage in active_stages.values():
            StageProcessor.close_stage_safely(remaining_stage)

        for item in final_metadata.attachments:
            choice.add_attachment(Attachment(**item.model_dump(exclude_none=True)))

        choice.set_state({
            _IS_GPA: True,
            _GPA_MESSAGES: final_metadata.state,
        })

        return Message(
            role=Role.ASSISTANT,
            content=StrictStr("".join(full_text)),
        )

    def __prepare_gpa_messages(self, request: Request, additional_instructions: Optional[str]) -> list[dict[str, Any]]:
        res_messages = []

        for idx in range(len(request.messages)):
            msg = request.messages[idx]
            if msg.role == Role.ASSISTANT:
                if msg.custom_content and msg.custom_content.state:
                    msg_state = msg.custom_content.state
                    if msg_state.get(_IS_GPA):
                        # 1. add user request (user message is always before assistant message)
                        res_messages.append(request.messages[idx - 1].dict(exclude_none=True))
                        # 2. Copy assistant message
                        copied_msg = deepcopy(msg)
                        copied_msg.custom_content.state = msg_state.get(_GPA_MESSAGES)
                        res_messages.append(copied_msg.dict(exclude_none=True))

        last_user_msg = request.messages[-1]
        custom_content = last_user_msg.custom_content
        if additional_instructions:
            res_messages.append(
                {
                    "role": Role.USER,
                    "content": f"{last_user_msg.content}\n\n{additional_instructions}",
                    "custom_content": custom_content.dict(exclude_none=True) if custom_content else None,
                }
            )
        else:
            res_messages.append(last_user_msg.dict(exclude_none=True))

