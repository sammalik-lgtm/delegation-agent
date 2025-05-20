import asyncio
import logging
import json
from typing import AsyncGenerator, List, DefaultDict, Annotated
from typing_extensions import override
from collections import defaultdict

from google.adk.agents import LlmAgent, BaseAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.events import Event
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def merge_async_generators(
    generators: List[AsyncGenerator[Event, None]]
) -> AsyncGenerator[Event, None]:
    """
    Merges multiple asynchronous generators into a single asynchronous generator.
    Events are yielded as they become available from any of the input generators.
    The order of events from within each individual generator is preserved.
    """
    if not generators:
        # If the list is empty, return an empty async generator
        if False: # This construct ensures it's an async generator type
            yield
        return

    queue = asyncio.Queue()
    # Using a unique object as a sentinel is safer than None,
    # in case any of your generators could legitimately yield None.
    _SENTINEL = object()

    async def _forward_generator_to_queue(gen: AsyncGenerator[Event, None], gen_idx: int):
        """
        Helper coroutine to read from one generator and put its items into the queue.
        Puts a sentinel value into the queue when the generator is exhausted or errors.
        """
        try:
            async for item in gen:
                await queue.put(item)
        except Exception as e:
            # Basic error logging. In a larger application, use the logging module.
            # The exception will also be collected by asyncio.gather in the finally block.
            print(f"[merge_async_generators] Error in source generator #{gen_idx}: {e!r}")
        finally:
            # Crucially, always put a sentinel to signal this generator is done.
            await queue.put(_SENTINEL)

    # Create a task for each generator to run it concurrently.
    producer_tasks = [
        asyncio.create_task(_forward_generator_to_queue(gen, i))
        for i, gen in enumerate(generators)
    ]

    finished_producer_count = 0
    try:
        while finished_producer_count < len(generators):
            item = await queue.get()
            if item is _SENTINEL:
                finished_producer_count += 1
            else:
                yield item
            queue.task_done()
    except asyncio.CancelledError:
        for task in producer_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*producer_tasks, return_exceptions=True)
        raise
    finally:
        results = await asyncio.gather(*producer_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                # Log unhandled exceptions from producer tasks.
                print(f"[merge_async_generators] Unhandled exception in producer task for generator #{i}: {result!r}")

def generate_delegator_prompt(agent_names: list[str], agent_descriptions: list[str], instruction: str) -> str:
    agents = "\n".join([f"* {name} -> {description}" for name, description in zip(agent_names, agent_descriptions)])

    return f"""
    You are a task delegation agent. Your responsibility is to analyze a user's request and determine which of the following agents should be assigned the task.

    Your available agents are: 
    {agents}

    You are tasked with analyzing a user prompt and assigning specific tasks to different agents.

    Your output should be a JSON-formatted collection of key/value pairs, where:
    *   The keys are the agents, using the following available agents.
    *   The values are the specific tasks to be given to that agent.
    *   If an agent does not have any tasks, its value should be an empty list.
    *   The output should be directly parsable by json.loads in python.
    *   Only include agents that are necessary for the task. 
    *   Do not repeat agent names.

    Examples:

    If your available agents are: A,B,C,D
    User Prompt: "I need A to summarize passage 1 and B to rephrase passage 2."
    Output: 
    {{
        'A': "Summarize passage 1.",
        'B': "Rephrase passage 2."
    }}

    If your available agents are: Waldo, Timmy, Foo
    User Prompt: "I need everyone to work on a summarization task except for Waldo"
    Output:
    {{
        'Timmy': "Summarization Task.",
        'Foo': "Summarization Task.",
    }}

    If your available agents are: Apple, Pear, Banana
    User Prompt: "You need to research six topics (A, B, C, D, E, and F), with a critical constraint: Topic E must be researched after Topic B. Split work accordingly to get these tasks done, adhering to the given sequential constraints."
    Output:
    {{
        'Apple': "Research A and D.",
        'Pear': "Research B, then research E.",
        'Banana': "Research C and F."

    }}

    Your specific role as delegator is the following: {instruction}
    """



def get_agent_names(agents: List[LlmAgent]) -> List[str]:
    return [agent.name for agent in agents]

def get_agent_descriptions(agents: List[LlmAgent]) -> List[str]:
    return [agent.description for agent in agents]


class DynamicDelegationAgent(BaseAgent):
    """
    Meta-agent that manages a pool of agents, by dynamically distributing tasks, 
    and collating the responses from each agent.
    """

    delegation_agent: LlmAgent
    agent_directory: DefaultDict[str, Annotated[LlmAgent, Field(default_factory=lambda: None)]]

    model_config = {"arbitrary_types_allowed": True}
    def __init__(
        self, 
        name: str, 
        delegator_role_prompt: str,
        agents: List[LlmAgent]
    ):

        delegation_agent = LlmAgent(
            name = "delegation_agent",
            model = "gemini-2.0-flash",
            description = "An agent to delegate tasks to other agents",
            instruction = generate_delegator_prompt(
                get_agent_names(agents), 
                get_agent_descriptions(agents),
                delegator_role_prompt
            ),
            output_key = "delegations"
        )

        agent_directory = defaultdict(str)
        for agent in agents:
            agent_directory[agent.name] = agent

        super().__init__(
            name=name,
            delegation_agent=delegation_agent,
            agent_directory=agent_directory
        )

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        logger.info("**Starting the Dynamic Delegation Agent**")

        logger.info("Running the main delegation agent...")
        async for event in self.delegation_agent.run_async(ctx):
            logger.info(f"[{self.name}] Event from DelegationAgent: {event.model_dump_json(indent=2, exclude_none=True)}")
            yield event
        
        logger.info("Validating the `delegations` key...")
        if "delegations" not in ctx.session.state or not ctx.session.state["delegations"]:
            logger.error(f"[{self.name}] Failed to generate delegation. Aborting workflow.")
            return # Stop processing if initial story failed.
        
        input_json_string = ctx.session.state.get('delegations').strip()
        if input_json_string.startswith("```json"):
          input_json_string = input_json_string[7:].strip()
        if input_json_string.endswith("```"):
          input_json_string = input_json_string[:-3].strip()

        agents_to_delegate = json.loads(input_json_string)
        logger.info(f"[{self.name}] Agents to delegate to : {agents_to_delegate}")
        
        sub_agent_async_generators = []
        for raw_delegation in agents_to_delegate:
            chosen_agent = self.agent_directory.get(raw_delegation.strip(), None)
            if not chosen_agent:
                logger.warning(f"{raw_delegation.strip()} is not an available agent!")
                continue
            sub_agent_async_generators.append(chosen_agent.run_async(ctx))
            
        async for event in merge_async_generators(sub_agent_async_generators):
            yield event
        