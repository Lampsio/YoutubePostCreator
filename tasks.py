from crewai import Task

class YoutubeTasks:

    def research_task(agent, video_url):
        return Task(description=(f"""
                Gather title, transcript, and key highlights from the YouTube video at {video_url}.
                Focus on extracting the main points that will attract viewers.
                """),
                agent=agent,
                expected_output=f"Describe sumamry video"
            )
    
    def post_creation_task(agent, summary_text , url):
        return Task(description=(f"""
                Create an engaging post based on the information gathered about the YouTube video at {summary_text}.
                The post should highlight the main points and encourage viewers to watch the video.
                Also use the channel name to greet

                When creating a post, make appropriate indentations in the text and start the text styling on a new line.

                When creating a post, write a short description of the film, why it should be watched, an invitation to discuss in the film's comments.

                Always add a link to the post that will redirect to the YouTube page with the video: {url}
                """),
                agent=agent,
                context=[summary_text],
                expected_output=f"Post for social media",
                output_file=f"output/post.txt"
            )