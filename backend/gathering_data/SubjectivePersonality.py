import csv
from math import ceil

import requests
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from tqdm import tqdm

ELEMENTS_ON_PAGE = 50
DOWNLOAD_DIR = "../../static/img"
DOWNLOAD_IMAGE = False
NEXT_PAGE_BUTTON_INDEX = 2
OUTPUT_FILE = "../../static/csv/records.csv"


class SubjectivePersonality:
    """Class used to get persons from subjectivepersonality.com with ops type and link to interview."""

    def __init__(self):
        """Initialize selenium driver."""
        self.driver = webdriver.Chrome()
        self.driver.get("https://app.subjectivepersonality.com/search?text=ops")

    def get_persons(self, limit: int) -> list:
        """Get persons from page with type and link to interview.

        :param limit: maximum number of persons to get
        :return: list of persons: [{'name', 'ops', 'image_url', 'profile_link', 'community_member', 'youtube_link'}]
        """
        total_iterations = int(ceil(limit / ELEMENTS_ON_PAGE))
        results = []

        next_page_button = self.__find_next_page_button()

        for _ in tqdm(range(total_iterations + 1), desc="Status"):
            type_records = self.__get_elements_from_page("type-record")

            # we need to skip every second element because there are duplicates
            elements_with_images = self.__get_elements_from_page("image-img")[::2]

            for record, image in zip(type_records, elements_with_images):

                if self.__check_speculation(record):
                    continue

                name, ops = self.__get_name_and_ops(record.text)

                if image.text == "Image Missing":
                    image_url = None
                else:
                    image_url = self.__get_image_url(image.get_attribute("outerHTML"))
                    name = name.replace("/", "%2F")  # ("%2F") represent a URL-encoded form of the forward slash ("/")
                    if DOWNLOAD_IMAGE:
                        self.download_image(name, image_url)

                result = {
                    "name": name,
                    "ops": ops,
                    "image_url": image_url,
                    "profile_link": f"http://app.subjectivepersonality.com/search/person?person={name}",
                    "community_member": self.__check_community_member(record),
                    "youtube_link": self.__get_youtube_link(record),
                }
                results.append(result)

            next_page_button.click()

        self.__save_results(results)
        return results

    def __get_elements_from_page(self, class_name: str) -> list:
        """Get class elements from the current page.

        :param class_name: class name to search for
        :return: list of elements found
        """
        self.driver.implicitly_wait(10)
        return self.driver.find_elements(By.CLASS_NAME, class_name)

    def __find_next_page_button(self):
        """Find and return the next page button element.

        :return: next page button element
        """
        buttons = self.__get_elements_from_page("btn-secondary")
        return buttons[NEXT_PAGE_BUTTON_INDEX]

    def __check_community_member(self, record):
        """Check if a person is a community member based on the record.

        :param record: the record to check
        :return: 1 if a community member, 0 otherwise
        """
        try:
            self.driver.implicitly_wait(0)
            record.find_element(By.CLASS_NAME, "community-member-logo")
            return 1
        except NoSuchElementException:
            return 0

    def __get_youtube_link(self, record):
        """Get the YouTube link from the record.

        :param record: the record to extract the link from
        :return: the YouTube link or None if not found
        """
        try:
            self.driver.implicitly_wait(0)
            return record.find_element(By.CLASS_NAME, "youtube-link").get_attribute("href")
        except NoSuchElementException:
            return None

    def __check_speculation(self, record):
        """Check if a record contains speculation tag (has a child with class 'ml-1').

        :param record: the record to check
        :return: True if speculation tag is present, False otherwise
        """
        try:
            self.driver.implicitly_wait(0)
            record.find_element(By.CLASS_NAME, "ml-1")
            return True
        except NoSuchElementException:
            return False

    @staticmethod
    def __get_image_url(text: str) -> str:
        """Extract and return the image URL from the given text.

        :param text: input text containing the image URL
        :return: extracted image URL
        """
        tag = "&quot;"
        start = text.find(tag)
        end = text.find(tag, start + 1)
        return text[start + len(tag):end]

    @staticmethod
    def __get_name_and_ops(text: str) -> tuple[str, str]:
        """Extract and return the name and ops from the given text.

        If the text does not contain a name or ops, we return empty strings.

        :param text: input text containing name and ops
        :return: tuple containing name and ops
        """
        result = text.split("\n")

        if len(result) == 1:
            return "", ""
        if result[0] != "Image Missing":
            return result[0], result[1]
        return result[1], result[2]

    @staticmethod
    def __save_results(records: list) -> None:
        """Save a list of records to a CSV file.

        :param records: list of records to be saved
        """
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
            csv_writer = csv.DictWriter(file, fieldnames=["name", "ops", "image_url", "profile_link", "community_member", "youtube_link"])
            csv_writer.writeheader()
            for record in records:
                csv_writer.writerow(record)
        print(f"Results saved in {OUTPUT_FILE}.")

    @staticmethod
    def download_image(name: str, url: str) -> None:
        """Download the image of a person.

        :param name: name of person from the image
        :param url: url of the image
        """
        data = requests.get(url).content
        with open(f"{DOWNLOAD_DIR}\\{name}.jpg", "wb") as f:
            f.write(data)


sp = SubjectivePersonality()
sp.get_persons(1900)
