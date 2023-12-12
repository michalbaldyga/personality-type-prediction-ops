from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from urllib.parse import urlparse, parse_qs


class VideoTranscriptFetcher:
    """
    Class for fetching video transcripts from YouTube URLs.
    """

    def __init__(self):
        """
        Initializes the VideoTranscriptFetcher object.

        :return: None
        """
        self.formatter = TextFormatter()

    def get_video_transcript(self, video_url: str) -> str:
        """
        Get the transcript of a video based on its URL.

        :param video_url: URL of the video on YouTube.
        :return: Formatted video transcript or an empty string if the transcript is not available.
        """
        video_id = self.get_video_id_from_url(video_url)
        if video_id:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text_formatted = self.formatter.format_transcript(transcript)
            return text_formatted
        else:
            return ""

    def process_records_file(self, input_file_path: str, output_file_path: str):
        """
        Process a file containing records, fetch video transcripts, and save the results to another file.

        :param input_file_path: Path to the input file with records, where columns are name and video URL.
        :param output_file_path: Path to the output file where names and video transcripts will be saved.
        :return: None
        """
        with open(input_file_path) as records_file:
            output = []
            lines = records_file.readlines()

            for line in lines:
                name, video_url = line.strip().split(',')
                transcript = self.get_video_transcript(video_url)

                output.append(f'{name},{transcript}')

            with open(output_file_path, 'w') as new_file:
                new_file.writelines(output)

    @staticmethod
    def get_video_id_from_url(video_url: str) -> str:
        """
        Get the video identifier (video_id) from a YouTube URL.

        :param video_url: URL of the video on YouTube.
        :return: Video identifier (video_id) or None if it cannot be obtained.
        """
        parsed_url = urlparse(video_url)
        query_params = parse_qs(parsed_url.query)
        video_id = query_params.get('v')
        return video_id[0] if video_id else None


if __name__ == "__main__":
    input_file_path = '../../static/csv/records_update.csv'
    output_file_path = '../../static/csv/transcripts.csv'

    transcript_fetcher = VideoTranscriptFetcher()
    transcript_fetcher.process_records_file(input_file_path, output_file_path)
