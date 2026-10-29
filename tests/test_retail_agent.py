from copy import deepcopy
from unittest.mock import patch

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
from tau2.domains.retail.environment import get_environment
from tau2.domains.mock.environment import get_environment as get_mock_environment


def test_retail_requests_serial_tools_after_each_result():
    environment = get_environment()
    arguments = {"temperature": 0, "parallel_tool_calls": True}
    agent = LLMAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm="gpt-4.1",
        llm_args=arguments,
    )
    before = deepcopy(arguments)
    initial_db = environment.tools.db.model_dump()
    address = {
        "address1": "250 Test Avenue",
        "address2": "Apt 4",
        "city": "Boston",
        "state": "MA",
        "country": "USA",
        "zip": "02110",
    }
    state = agent.get_init_state()
    calls = [
        ToolCall(
            id="profile",
            name="modify_user_address",
            arguments={"user_id": "ivan_santos_6635", **address},
        ),
        ToolCall(
            id="order",
            name="modify_pending_order_address",
            arguments={"order_id": "#W8770097", **address},
        ),
    ]
    outputs = [AssistantMessage(role="assistant", tool_calls=[call]) for call in calls]
    incoming = UserMessage(role="user", content="Yes, update both addresses as listed.")
    first_result = None
    with patch("tau2.agent.llm_agent.generate", side_effect=outputs) as generate:
        for call in calls:
            message, state = agent.generate_next_message(incoming, state)
            assert message.tool_calls == [call]
            assert generate.call_args.kwargs["parallel_tool_calls"] is False
            if first_result is not None:
                assert first_result in generate.call_args.kwargs["messages"]
            incoming = environment.get_response(call)
            assert not incoming.error
            first_result = incoming
    assert arguments == before
    assert agent.llm_args == before
    expected = deepcopy(initial_db)
    expected["users"]["ivan_santos_6635"]["address"] = address
    expected["orders"]["#W8770097"]["address"] = address
    assert environment.tools.db.model_dump() == expected


def test_other_domains_keep_existing_generation_options():
    environment = get_mock_environment()
    agent = LLMAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm="gpt-4.1",
        llm_args={"temperature": 0},
    )
    with patch(
        "tau2.agent.llm_agent.generate",
        return_value=AssistantMessage(role="assistant", content="How can I help?"),
    ) as generate:
        agent.generate_next_message(
            UserMessage(role="user", content="Hello"), agent.get_init_state()
        )
    assert "parallel_tool_calls" not in generate.call_args.kwargs
