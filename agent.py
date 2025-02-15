import sys
import asyncio

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, SecretStr

# Import your browser and LLM modules
from browser_use import Agent, Browser, BrowserConfig
from langchain_google_genai import ChatGoogleGenerativeAI

app = FastAPI()

# Configurations (adjust as needed)
CHROME_PATH = r"/usr/bin/google-chrome-stable"
MODEL_NAME = 'gemini-2.0-flash-exp'
API_KEY = SecretStr("AIzaSyC59TDceV-5GBV7KjE_7cOZqzdZMlGx3I0")

# Pydantic model for the request body
class TaskRequest(BaseModel):
    task: str

@app.post("/task")
async def execute_task(task_request: TaskRequest):
    try:
        # Create a headless browser instance
        browser = Browser(
            config=BrowserConfig(
                chrome_instance_path=CHROME_PATH,
                headless=True,
            )
        )

        # Create the language model instance
        llm = ChatGoogleGenerativeAI(model=MODEL_NAME, api_key=API_KEY)

        # Create the agent with the task from the request body
        agent = Agent(
            task=task_request.task,
            llm=llm,
            browser=browser,
        )

        # Run the agent asynchronously
        result = await agent.run()

        # Close the browser after task completion
        await browser.close()

        return {"status": "success", "task": result.final_result(), "message": "Task executed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
