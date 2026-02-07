import json
from copy import deepcopy
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, Stage
from pydantic import StrictStr

from task.coordination.gpa import GPAGateway
from task.coordination.ums_agent import UMSAgentGateway
from task.logging_config import get_logger
from task.models import CoordinationRequest, AgentName
from task.prompts import COORDINATION_STRATEGY_PROMPT, SYNTHESIS_OUTPUT_PROMPT
from task.stage_util import StageProcessor

logger = get_logger(__name__)


class MASCoordinator:

    def __init__(self, endpoint: str, deployment_name: str, ums_agent_endpoint: str):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.ums_agent_endpoint = ums_agent_endpoint

    async def handle_request(self, choice: Choice, request: Request) -> Message:

        session_params = {
            "base_url": self.endpoint,
            "api_key": request.api_key,
            "api_version": '2025-01-01-preview'
        }
        api_client = AsyncDial(**session_params)

        logger.info(f"Starting multi-agent flow [ID: {request.id if hasattr(request, 'id') else 'N/A'}]")

        dispatch_stage = StageProcessor.open_stage(choice, "Route Discovery")
        try:
            execution_plan = await self.__prepare_coordination_request(
                client=api_client,
                request=request,
            )

            plan_raw = execution_plan.model_dump_json(indent=2)
            dispatch_stage.append_content(f"### Target Plan\n```json\n{plan_raw}\n```\n")
            logger.debug(f"Plan created for agent: {execution_plan.agent_name}")
        finally:
            StageProcessor.close_stage_safely(dispatch_stage)

        runtime_stage = StageProcessor.open_stage(choice, f"Agent Execution: {execution_plan.agent_name}")
        try:
            logger.info(f"Delegating task to {execution_plan.agent_name}...")
            delegated_result = await self.__handle_coordination_request(
                coordination_request=execution_plan,
                choice=choice,
                stage=runtime_stage,
                request=request,
            )
        except Exception as err:
            logger.exception(f"Agent execution failed: {execution_plan.agent_name}")
            raise err
        finally:
            StageProcessor.close_stage_safely(runtime_stage)

        logger.info("Synthesizing final response for user")
        return await self.__final_response(
            client=api_client,
            request=request,
            choice=choice,
            agent_message=delegated_result,
        )

    async def __prepare_coordination_request(self, client: AsyncDial, request: Request) -> CoordinationRequest:
        logger.debug("Generating coordination schema for request...")

        payload = {
            "messages": self.__prepare_messages(request, COORDINATION_STRATEGY_PROMPT),
            "deployment_name": self.deployment_name,
            "extra_body": {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "coordination_schema",
                        "schema": CoordinationRequest.model_json_schema()
                    }
                }
            }
        }

        completion = await client.chat.completions.create(**payload)
        raw_content = completion.choices[0].message.content

        return CoordinationRequest.model_validate_json(raw_content)

    def __prepare_messages(self, request: Request, system_prompt: str) -> list[dict[str, Any]]:
        formatted_history = [{"role": Role.SYSTEM, "content": system_prompt}]

        for m in request.messages:
            if m.role == Role.USER and m.custom_content:
                formatted_history.append({
                    "role": Role.USER,
                    "content": str(m.content)
                })
            else:
                formatted_history.append(m.model_dump(exclude_none=True))

        return formatted_history

    async def __handle_coordination_request(
            self,
            coordination_request: CoordinationRequest,
            choice: Choice,
            stage: Stage,
            request: Request
    ) -> Message:
        if coordination_request.agent_name is AgentName.GPA:
            return await GPAGateway(endpoint=self.endpoint).response(
                choice=choice,
                request=request,
                stage=stage,
                additional_instructions=coordination_request.additional_instructions,
            )

        elif coordination_request.agent_name is AgentName.UMS:
            return await UMSAgentGateway(ums_agent_endpoint=self.ums_agent_endpoint).response(
                choice=choice,
                request=request,
                stage=stage,
                additional_instructions=coordination_request.additional_instructions,
            )
        else:
            raise ValueError("Unknown Agent Name")

    async def __final_response(self, client: AsyncDial, choice: Choice, request: Request,
                               agent_message: Message) -> Message:
        msgs = self.__prepare_messages(request, SYNTHESIS_OUTPUT_PROMPT)

        updated_user_request = f"## CONTEXT:\n {agent_message.content}\n ---\n ## USER_REQUEST: \n {msgs[-1]["content"]}"
        msgs[-1]["content"] = updated_user_request

        chunks = await client.chat.completions.create(
            stream=True,
            messages=msgs,
            deployment_name=self.deployment_name
        )

        content = ''
        async for chunk in chunks:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    choice.append_content(delta.content)
                    content += delta.content

        return Message(
            role=Role.ASSISTANT,
            content=StrictStr(content),
            custom_content=agent_message.custom_content
        )
