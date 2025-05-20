import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from .dynamic_delegation_agent import DynamicDelegationAgent


def generate_agent_prompt(agent_letter: str) -> str:
    """
    Generates a prompt for an LLM, allowing for a customizable agent identifier.

    Args:
        agent_letter: The letter to use as the agent identifier (e.g., "A", "B", "X").
        context: A dictionary containing the context, including a 'delegations' key.

    Returns:
        A string containing the generated prompt.
        Returns an error message if the context is not valid.
    """

    if not isinstance(agent_letter, str) or len(agent_letter) != 1:
      return "Error: agent_letter must be a single character string."

    agent_name = f"agent {agent_letter}"
    prompt = f"""You are {agent_name}. Your job is to perform tasks if they are assigned to you from the 'delegations' key in the context. Don't execute the tasks; only repeat the task you were assigned and state your name in a friendly manner. ONLY READ YOUR INSTRUCTION AND THATS IT."""
    return prompt

a_agent = Agent(
    name = "risk_agent",
    model = "gemini-2.0-flash",
    description = "Stock Risk Analyst",
    instruction = generate_agent_prompt('A'),
    output_key = "A_output"
)

b_agent = Agent(
    name = "volatility_agent",
    model = "gemini-2.0-flash",
    description = "Stock Volatility Analyst",
    instruction = generate_agent_prompt('B'),
    output_key = "B_output"
)

c_agent = Agent(
    name = "chef_agent",
    model = "gemini-2.0-flash",
    description = "Chef",
    instruction = generate_agent_prompt('C'),
    output_key = "C_output"
)

d_agent = Agent(
    name = "writer_agent",
    model = "gemini-2.0-flash",
    description = "Writer",
    instruction = generate_agent_prompt('D'),
    output_key = "D_output"
)

root_agent = DynamicDelegationAgent(
    "root_agent",
    "You are the delegation agent for an investment firm that specializes in the food and book industries. Send tasks to agents appropriate for researching.",
    [a_agent, b_agent, c_agent, d_agent]
)