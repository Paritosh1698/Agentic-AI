from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

#Web Search Agent
web_search_agent = Agent(
    name="Web serach agent",
    role="Search the web for information",
    model=Groq(id="llama-3.2-90b-vision-preview"),
    tools=[DuckDuckGo()],
    instructions="Always include souorces",
    show_tool_calls=True,
    markdown=True,
)

#Financial Agent
finance_agent=Agent(
    name="Financial AI Agent",
    model=Groq(id="llama-3.2-90b-vision-preview"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True, company_info=True),
    ],
    instructions="Use tables to display the data",
    show_tool_calls=True,
    markdown=True,
)

multi_modal_agent=Agent(
    model=Groq(id="llama-3.2-90b-vision-preview"),
    team=[web_search_agent, finance_agent],
    instructions=["Always include sources","Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

multi_modal_agent.print_response("Summarize analyst recommendation and share the latest news for BA", stream=True)
