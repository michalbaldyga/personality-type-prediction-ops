"""
Download needed libraries using:
pip install --upgrade google-api-python-client
pip install --upgrade google-auth-oauthlib google-auth-httplib2
"""

import re
import requests

DURATION_LIMIT = 30


class YoutubeBrowser:
    def __init__(self):
        self.youtube_url = 'https://www.youtube.com/'
        self.watch_url = f'{self.youtube_url}watch?v='
        self.search_url = f'{self.youtube_url}results?search_query='
        self.length_filter = '&sp=EgIYAg%253D%253D'

    def get_interview_by_person(self, person):
        page_url = f"{self.search_url}{'+'.join(person.split())}+interview{self.length_filter}"
        return self.get_interview_by_search_link(page_url, person)

    def get_interview_by_search_link(self, page_url, person):
        page = requests.get(page_url)
        content = page.content.decode()
        urls = re.findall('/watch\?v=[^"\s]+', content)

        unique_urls = set()
        for url in urls:
            u = url.split("\\")
            unique_urls.add(u[0])

        index_limit = min(len(unique_urls), 30)

        for video_id in list(unique_urls)[:index_limit]:
            video_url = f"{self.youtube_url}{video_id}"
            if self.__is_interview_proper(video_url, person):
                return video_url

        return None

    def __is_interview_proper(self, video_url, person):
        page = requests.get(video_url)
        content = page.content.decode()

        title = self.__get_video_title(content)

        if not self.__is_name_in_title(person, title):
            return False
        if not self.__is_keyword_in_title(title):
            return False

        return True

    @staticmethod
    def __get_video_title(content):
        title_tag = re.findall('<title>[^<]+</title>', content)
        title = title_tag[0][7:-8]
        return title

    @staticmethod
    def __is_name_in_title(name, title):
        parts = name.lower().split()
        lowered_title = title.lower()
        if parts[0] in lowered_title and parts[-1] in lowered_title:
            return True

        return False

    @staticmethod
    def __is_keyword_in_title(title):
        lowered_title = title.lower()
        keywords = ['interview', 'podcast', 'conversation', 'discussion']

        for keyword in keywords:
            if keyword in lowered_title:
                return True

        return False
