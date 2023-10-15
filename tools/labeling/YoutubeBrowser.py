"""
Download needed libraries using:
pip install --upgrade google-api-python-client
pip install --upgrade google-auth-oauthlib google-auth-httplib2
"""

from googleapiclient.discovery import build
import re
import requests

API_KEY = 'AIzaSyAXSg_4raL8Mzi0p4y31rQdeJzdEBVZw2E'
DURATION_LIMIT = 30


class YoutubeBrowser:
    def __init__(self):
        self.youtube = build('youtube', 'v3', developerKey=API_KEY)
        self.youtube_url = 'https://www.youtube.com/'
        self.watch_url = f'{self.youtube_url}watch?v='
        self.search_url = f'{self.youtube_url}results?search_query='
        self.length_filter = '&sp=EgIYAg%253D%253D'

    def get_interview(self, person):
        page_url = f"{self.search_url}{'+'.join(person.split())}+interview{self.length_filter}"
        page = requests.get(page_url)
        content = page.content.decode()
        urls = re.findall('/watch\?v=[^"\s]+', content)

        unique_urls = set()
        for url in urls:
            u = url.split("\\")
            unique_urls.add(u[0])

        for video_id in unique_urls:
            video_url = f"{self.youtube_url}{video_id}"
            if self.__is_interview_proper(video_url, person):
                return video_url

    def __is_interview_proper(self, video_url, person):
        page = requests.get(video_url)
        content = page.content.decode()

        title = self.__get_video_title(content)
        duration = self.__get_video_duration(content)

        if person in title and duration > DURATION_LIMIT:
            return True
        else:
            return False

    def __get_video_title(self, content):
        title_tag = re.findall('<title>[^<]+</title>', content)
        title = title_tag[0][7:-8]
        return title

    def __get_video_duration(self, content):
        duration_tag = re.findall('meta itemprop="duration" content="PT[0-9A-Z]+', content)
        duration = duration_tag[0].split('"')[-1]
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


ytb = YoutubeBrowser()
with open('records.csv') as records_file:
    output = []
    lines = records_file.readlines()

    for l in lines[1:]:
        line = l[:-1]
        person = line.split(',')[0]

        try:
            interview_link = ytb.get_interview(person)
            print(f"{len(output)}: {interview_link}")
            if interview_link:
                output.append(f'{line},{interview_link}')
            else:
                output.append(f'{line},')
        except:
            break

    with open('../../static/csv/records_update.csv', 'w') as new_file:
        print(output)
        new_file.writelines(output)
