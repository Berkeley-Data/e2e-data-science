import json
from typing import Any, Callable, Generator, Optional
from uuid import uuid4

import backoff
import mlflow
import openai
from databricks.sdk import WorkspaceClient
from databricks_openai import UCFunctionToolkit, VectorSearchRetrieverTool
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from openai import OpenAI
from pydantic import BaseModel
from unitycatalog.ai.core.base import get_uc_function_client

############################################
# Define your LLM endpoint and system prompt
############################################
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"

SYSTEM_PROMPT = """Act as an assistant for wind turbine maintenance technicians.
    These are the tools you can use to answer questions:
      - turbine_maintenance_predictor: takes as input sensor_readings and predicts whether or not a turbine is at risk of failure.
        Use turbine_specifications_retriever to get the current status.
      - turbine_maintenance_guide_retriever: send the question, model and error code to get the relevant turbine part maintenance guide to assist the user with maintenance operation.
      - turbine_specifications_retriever: takes turbine_id as input and retrieves turbine specifications.
    

If a user gives you a turbine ID, first look up that turbine's information with turbine_specifications_retriever. 
If a user asks for recommendations on how to do maintenance on a turbine, use the maintenance guide to search how to maintain the parts and provide guidance on the steps required to fix the turbines"""


###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/generative-ai/agent-framework/agent-tool.html
###############################################################################
class ToolInfo(BaseModel):
    """
    Class representing a tool for the agent.
    - "name" (str): The name of the tool.
    - "spec" (dict): JSON description of the tool (matches OpenAI Responses format)
    - "exec_fn" (Callable): Function that implements the tool logic
    """

    name: str
    spec: dict
    exec_fn: Callable


def create_tool_info(tool_spec, exec_fn_param: Optional[Callable] = None):
    tool_spec["function"].pop("strict", None)
    tool_name = tool_spec["function"]["name"]
    udf_name = tool_name.replace("__", ".")

    # Define a wrapper that accepts kwargs for the UC tool call,
    # then passes them to the UC tool execution client
    def exec_fn(**kwargs):
        function_result = uc_function_client.execute_function(udf_name, kwargs)
        if function_result.error is not None:
            return function_result.error
        else:
            return function_result.value
    return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn_param or exec_fn)


TOOL_INFOS = []

# You can use UDFs in Unity Catalog as agent tools
# TODO: Add additional tools
UC_TOOL_NAMES = ["main.dbdemos_iot_turbine.turbine_maintenance_guide_retriever", "main.dbdemos_iot_turbine.turbine_maintenance_predictor", "main.dbdemos_iot_turbine.turbine_specifications_retriever"]

uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
uc_function_client = get_uc_function_client()
for tool_spec in uc_toolkit.tools:
    TOOL_INFOS.append(create_tool_info(tool_spec))


# Use Databricks vector search indexes as tools
# See [docs](https://docs.databricks.com/generative-ai/agent-framework/unstructured-retrieval-tools.html) for details

# # (Optional) Use Databricks vector search indexes as tools
# # See https://docs.databricks.com/generative-ai/agent-framework/unstructured-retrieval-tools.html
# # for details
VECTOR_SEARCH_TOOLS = []
# # TODO: Add vector search indexes as tools or delete this block
# VECTOR_SEARCH_TOOLS.append(
#         VectorSearchRetrieverTool(
#         index_name="",
#         # filters="..."
#     )
# )



class ToolCallingAgent(ResponsesAgent):
    """
    Class representing a tool-calling Agent
    """

    def __init__(self, llm_endpoint: str, tools: list[ToolInfo]):
        """Initializes the ToolCallingAgent with tools."""
        self.llm_endpoint = llm_endpoint
        self.workspace_client = WorkspaceClient()
        self.model_serving_client: OpenAI = (
            self.workspace_client.serving_endpoints.get_open_ai_client()
        )
        # internal state will be standard chat completion messages
        self.messages: list[dict[str, Any]] = None
        self._tools_dict = {tool.name: tool for tool in tools}

    def get_tool_specs(self) -> list[dict]:
        """Returns tool specifications in the format OpenAI expects."""
        return [tool_info.spec for tool_info in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Executes the specified tool with the given arguments."""
        return self._tools_dict[tool_name].exec_fn(**args)

    def convert_to_chat_completion_format(self, message: dict[str, Any]) -> dict[str, Any]:
        """Convert from Responses API to be compatible with a ChatCompletions LLM endpoint"""
        msg_type = message.get("type", None)
        if msg_type == "function_call":
            return [
                {
                    "role": "assistant",
                    "content": "tool call",
                    "tool_calls": [
                        {
                            "id": message["call_id"],
                            "type": "function",
                            "function": {
                                "arguments": message["arguments"],
                                "name": message["name"],
                            },
                        }
                    ],
                }
            ]
        elif msg_type == "message" and isinstance(message["content"], list):
            return [
                {"role": message["role"], "content": content["text"]}
                for content in message["content"]
            ]
        elif msg_type == "function_call_output":
            return [
                {
                    "role": "tool",
                    "content": message["output"],
                    "tool_call_id": message["call_id"],
                }
            ]
        compatible_keys = ["role", "content", "name", "tool_calls", "tool_call_id"]
        if not message.get("content") and message.get("tool_calls"):
            message["content"] = "tool call"
        filtered = {k: v for k, v in message.items() if k in compatible_keys}
        return [filtered] if filtered else []

    def prepare_messages_for_llm(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter out message fields that are not compatible with LLM message formats and convert from Responses API to ChatCompletion compatible"""
        chat_msgs = []
        for msg in messages:
            chat_msgs.extend(self.convert_to_chat_completion_format(msg))
        return chat_msgs

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    @mlflow.trace(span_type=SpanType.LLM)
    def call_llm(self) -> dict[str, Any]:
        return (
            self.model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=self.prepare_messages_for_llm(self.messages),
                tools=self.get_tool_specs(),
            )
            .choices[0]
            .message.to_dict()
        )

    def handle_tool_call(
        self, tool_calls: list[dict[str, Any]]
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """
        Execute tool calls, add them to the running message history, and return a ResponsesStreamEvent w/ tool output
        """
        for tool_call in tool_calls:
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_function_call_item(
                    str(uuid4()),
                    tool_call["id"],
                    tool_call["function"]["name"],
                    tool_call["function"]["arguments"],
                ),
            )
            function = tool_call["function"]
            args = json.loads(function["arguments"])
            # Cast tool result to a string, since not all tools return as tring
            result = str(self.execute_tool(tool_name=function["name"], args=args))
            self.messages.append(
                {"role": "tool", "content": result, "tool_call_id": tool_call["id"]}
            )
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_function_call_output_item(
                    tool_call["id"],
                    result,
                ),
            )

    def call_and_run_tools(
        self,
        max_iter: int = 10,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        for _ in range(max_iter):
            last_msg = self.messages[-1]
            if tool_calls := last_msg.get("tool_calls", None):
                yield from self.handle_tool_call(tool_calls)
            elif last_msg.get("role", None) == "assistant":
                return
            else:
                llm_output = self.call_llm()
                self.messages.append(llm_output)

                if not llm_output.get("tool_calls"):
                    yield ResponsesAgentStreamEvent(
                        type="response.output_item.done",
                        item=self.create_text_output_item(llm_output["content"], str(uuid4())),
                    )

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item("Max iterations reached. Stopping.", str(uuid4())),
        )

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        self.messages = self.prepare_messages_for_llm([i.model_dump() for i in request.input])
        if SYSTEM_PROMPT:
            self.messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        yield from self.call_and_run_tools()


# Log the model using MLflow
mlflow.openai.autolog()
AGENT = ToolCallingAgent(llm_endpoint=LLM_ENDPOINT_NAME, tools=TOOL_INFOS)
mlflow.models.set_model(AGENT)
