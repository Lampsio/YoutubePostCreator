import os
from langchain_groq import ChatGroq
from crewai import Agent
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

class YouTubeVideoAgents():
    def __init__(self) -> None:
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model="gemma2-9b-it"
        )
    
    def youtube_analysis(self):
        return Agent(
            role="YouTube Analysis",
            goal=f"""
                Write a summary of this video without revealing all the data
                Analyze the transcript of the video from the YouTube platform,
                extract the topic of the video from the transcript and the topics discussed in the video that
                encourage the viewer to click on the video.
            """,
            backstory="""
            As an Youtube Analysis, you are responsible for analyzing the video transcript.
            """,
            verbose=True,
            llm=self.llm
        )
    
    def post_creator(self):
        return Agent(
            role="Post Creator",
            goal=f"""
                Using informal language, transform the summary of the video into a post on discord or youtube.
                Encourage regular and new viewers to watch the video without giving away all the information
                Focus on extracting the main points that will attract viewers.
            """,
            backstory="""
            As an Post Creator, you are responsible for creating post for social media.
            """,
            verbose=True,
            llm=self.llm
        )
    