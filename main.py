from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from agents import YouTubeVideoAgents
from tasks import YoutubeTasks
from crewai import Crew
import os
import time
import agentops
from dotenv import load_dotenv
load_dotenv()

agentops_api_key = os.getenv('AGENTOPS_API_KEY')

agentops.init(api_key=agentops_api_key)

def load_youtube_info(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True,
    language=["en", "id"],
    translation="en",)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = text_splitter.split_documents(documents)
    return all_splits[:5]

url = input("Podaj url video: ")
url_data = load_youtube_info(url)
print(url_data)

agents = YouTubeVideoAgents()

youtube_analysis = agents.youtube_analysis()
post_creator = agents.post_creator()

tasks = YoutubeTasks

summary_youtube_task = tasks.research_task(
    agent=youtube_analysis,
    video_url = url_data
)

post_youtube_task = tasks.post_creation_task(
    agent = post_creator,
    summary_text = summary_youtube_task,
    url = url
)

crew = Crew(
    agents=[youtube_analysis, post_creator],
    tasks=[
        summary_youtube_task,
        post_youtube_task
    ],
    max_rpm=29
)

start_time = time.time()

results = crew.kickoff()

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Crew kickoff took {elapsed_time} seconds.")
print(f"Crew usage", crew.usage_metrics)

agentops.end_session("Success")

