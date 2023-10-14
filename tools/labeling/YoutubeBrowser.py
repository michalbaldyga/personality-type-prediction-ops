"""
Download needed libraries using:
pip install --upgrade google-api-python-client
pip install --upgrade google-auth-oauthlib google-auth-httplib2
"""

from googleapiclient.discovery import build
import re

API_KEY = 'AIzaSyAXSg_4raL8Mzi0p4y31rQdeJzdEBVZw2E'
DURATION_LIMIT = 30


class YoutubeBrowser:
    def __init__(self):
        self.youtube = build('youtube', 'v3', developerKey=API_KEY)

    def get_interview(self, person):
        query = f'{person} interview'
        request = self.youtube.search().list(
            q=query,
            type='video',
            part='id',
            maxResults=10,
            videoDuration='long'
        )
        response = request.execute()

        video_ids = [item['id']['videoId'] for item in response['items']]
        for video_id in video_ids:
            if self.__is_interview_proper(video_id, person):
                return f'https://www.youtube.com/watch?v={video_id}'

    def __is_interview_proper(self, video_id, person):
        title = self.__get_video_title(video_id)
        duration = self.__get_video_duration(video_id)

        if person in title and duration > DURATION_LIMIT:
            return True
        else:
            return False

    def __get_video_title(self, video_id):
        request = self.youtube.videos().list(
            part='snippet',
            id=video_id
        )

        response = request.execute()
        video = response['items'][0]
        return video['snippet']['title']

    def __get_video_duration(self, video_id):
        request = self.youtube.videos().list(
            part='contentDetails',
            id=video_id
        )

        response = request.execute()
        video = response['items'][0]
        duration = video['contentDetails']['duration']
        return self.__convert_duration_to_minutes(duration)

    @staticmethod
    def __convert_duration_to_minutes(duration):
        hours_match = re.search(r'(\d+)H', duration)
        minutes_match = re.search(r'(\d+)M', duration)

        # Initialize variables for each component (hours, minutes)
        hours = int(hours_match.group(1)) if hours_match else 0
        minutes = int(minutes_match.group(1)) if minutes_match else 0

        # Calculate the total duration in minutes
        total_minutes = (hours * 60) + minutes
        return total_minutes
