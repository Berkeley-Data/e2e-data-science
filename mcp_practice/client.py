import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
import openai

# # from anthropic import Anthropic
# from dotenv import load_dotenv

# load_dotenv()  # load environment variables from .env

OPENAI_API_KEY = ""
openai.api_key = OPENAI_API_KEY  # Replace with your API key

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openAI_client = openai

    # methods will go here


    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])


    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""

        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()

        available_tools = [{
            "type": 'function',
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        } for tool in response.tools]


        response = self.openAI_client.responses.create(
                model="gpt-4",  # or "gpt-3.5-turbo" if preferred
                input=messages,
                tools=available_tools

            )
        

        # Process response and handle tool calls
        final_text = []

        tool_call_id = response.output[0].call_id
        for content in response.output:

            if content.type == 'message':
                final_text.append(content.content[0].text)

            elif content.type == 'function_call':
                # tool_call_id = content.call_id
                tool_name = content.name
                tool_arg_str = content.arguments
                tool_args = json.loads(tool_arg_str)

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                
                if hasattr(result.content, 'text'):
                    tool_result_text = result.content.text
                          
                elif hasattr(result.content, 'model_dump'):
                    dumped = result.content.model_dump()
                    tool_result_text = dumped.get('text', str(dumped))
                else:
                    tool_result_text = str(result.content[0].text)
            
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")


                messages.append({
                    "type": "function_call",
                    "call_id": tool_call_id,
                    "name": tool_name,
                    "arguments":tool_arg_str
                })

                messages.append({
                    "type": "function_call_output",
                    "call_id": tool_call_id,
                    "output": tool_result_text
                })


                response = self.openAI_client.responses.create(
                    model= "gpt-3.5-turbo", # if preferred
                    input=messages,
                    tools=available_tools

                )
                final_text.append(response.output[0].content[0].text)

        return "\n".join(final_text)


    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
