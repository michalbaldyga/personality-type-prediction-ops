""" Class used to get persons from subjectivepersonality.com with ops type and link to interview """
from typing import Tuple
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from tqdm import tqdm
from math import ceil
from image_downloader import download_image

ELEMENTS_ON_PAGE = 50


class SubjectivePersonality:
    def __init__(self):
        """ Initialize selenium driver """
        self.driver = webdriver.Chrome()
        self.driver.get("https://app.subjectivepersonality.com/search?text=ops")

    def get_persons(self, limit: int) -> list:
        """ Get persons from page with type and link to interview

        :param limit: maximum number of persons to get
        :return: list of persons: [(ops type, link to interview)]
        """
        total_iterations = int(ceil(limit / ELEMENTS_ON_PAGE))
        results = []

        next_page_button = self.__find_next_page_button()

        for _ in tqdm(range(total_iterations), desc='Status'):
            elements_with_text = self.__get_elements_from_page('type-record')
            elements_with_images = self.__get_elements_from_page('image-img')

            # we need to skip every second element because there are duplicates
            elements_with_images = elements_with_images[::2]

            for text, image in zip(elements_with_text, elements_with_images):
                if image.text == 'Image Missing':
                    image_url = None
                    name, ops = self.__get_name_and_ops(text.text, True)
                else:
                    image_url = self.__get_image_url(image.get_attribute("outerHTML"))
                    name, ops = self.__get_name_and_ops(text.text, False)
                    name = name.replace('/', ' or ')
                    download_image(name, image_url)
                results.append({'name': name, 'ops': ops, 'image_url': image_url})

            next_page_button.click()

        self.__save_results(results)
        return results

    def __get_elements_from_page(self, class_name: str) -> list:
        """ Get class elements from current page

        :param class_name: class name to search for
        :return: list of elements found
        """
        self.driver.implicitly_wait(10)
        return self.driver.find_elements(By.CLASS_NAME, class_name)

    def __find_next_page_button(self):
        """
        Find and return the next page button element.

        :return: next page button element
        """
        buttons = self.__get_elements_from_page('btn-secondary')
        next_page_button = buttons[2]
        return next_page_button

    @staticmethod
    def __get_image_url(text: str) -> str:
        """
        Extract and return the image URL from the given text.

        :param text: input text containing the image URL
        :return: extracted image URL
        """
        tag = '&quot;'
        start = text.find(tag)
        end = text.find(tag, start + 1)
        result = text[start + len(tag):end]
        return result

    @staticmethod
    def __get_name_and_ops(text: str, is_image_missing: bool) -> Tuple[str, str]:
        """
        Extract and return the name and ops from the given text.

        :param text: input text containing name and ops
        :param is_image_missing: boolean indicating whether image is missing
        :return: tuple containing name and ops
        """
        result = text.split('\n')
        if is_image_missing:
            return result[1], result[2]
        else:
            return result[0], result[1]

    @staticmethod
    def __save_results(records: list):
        """
         Save a list of records to a CSV file.

         :param records: list of records to be saved
         """
        csv_file = 'records.csv'
        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            csv_writer = csv.DictWriter(file, fieldnames=['name', 'ops', 'image_url'])
            csv_writer.writeheader()
            for record in records:
                csv_writer.writerow(record)
        print(f'Results saved in {csv_file}.')


sp = SubjectivePersonality()
sp.get_persons(1800)
