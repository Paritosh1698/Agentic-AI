import openai
from phi.agent import Agent
import phi.api
from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.playground import Playground, serve_playground_app

#phi.api = os.getenv("PHI_API_KEY")

#Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=Groq(id="llama-3.2-90b-vision-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include souorces"],
    show_tool_calls=True,
    markdown=True,
)

#Financial Agent
finance_agent = Agent(
    name="Financial AI Agent",
    model=Groq(id="llama-3.2-90b-vision-preview"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True, company_info=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

app = Playground(agents=[web_search_agent,finance_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app", reload = True)
