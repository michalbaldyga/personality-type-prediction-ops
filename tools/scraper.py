from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from urllib.parse import urlparse, parse_qs


def get_video_id_from_url(video_url: str) -> str:
    # Parse the URL to get its components
    parsed_url = urlparse(video_url)

    # Extract the query parameters from the URL
    query_params = parse_qs(parsed_url.query)

    # Get the 'v' parameter, which represents the video_id
    video_id = query_params.get('v')

    return video_id[0]


def get_video_transcript(video_url: str) -> str:
    # Extract the video ID from the provided URL
    video_id = get_video_id_from_url(video_url)

    # Use the YouTubeTranscriptApi to retrieve the transcript for the video
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    # Initialize a TextFormatter to format the transcript
    formatter = TextFormatter()

    # Format the transcript using the TextFormatter
    text_formatted = formatter.format_transcript(transcript)

    return text_formatted
