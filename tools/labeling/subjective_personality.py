""" Class used to get persons from subjectivepersonality.com with ops type and link to interview """
from time import sleep
from typing import Tuple
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By

ELEMENTS_ON_PAGE = 50


def get_image_url(text: str) -> str:
    tag = '&quot;'
    start = text.find(tag)
    end = text.find(tag, start+1)
    result = text[start+len(tag):end]
    return result


def get_name_and_ops(text: str, is_image_missing: bool) -> Tuple[str, str]:
    result = text.split('\n')
    if is_image_missing:
        return result[1], result[2]
    else:
        return result[0], result[1]


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
        current = 0
        results = []
        #show_types_button = self.driver.find_element(By.CLASS_NAME, 'btn-sm')
        #show_types_button.click()
        sleep(5)
        next_page_button = self.__find_next_page_button()

        while current < limit:
            sleep(5)
            elements_with_text = self.__get_elements_from_page('type-record')
            elements_with_images = self.__get_elements_from_page('image-img')

            # we need to skip every second element
            elements_with_images = elements_with_images[::2]

            for text, image in zip(elements_with_text, elements_with_images):
                if image.text == 'Image Missing':
                    image_url = None
                    name, ops = get_name_and_ops(text.text, True)
                else:
                    image_url = get_image_url(image.get_attribute("outerHTML"))
                    name, ops = get_name_and_ops(text.text, False)
                results.append({'name': name, 'ops': ops, 'image_url': image_url})

            next_page_button.click()
            current += 50
            print(f'Status: {current}/{limit}')

        self.__save_result(results)
        return results

    def __get_elements_from_page(self, class_name: str) -> list:
        """ Get class elements from current page
        :param class_name: class name
        :return: list of elements
        """
        return self.driver.find_elements(By.CLASS_NAME, class_name)

    def __find_next_page_button(self):
        buttons = self.__get_elements_from_page('btn-secondary')
        next_page_button = buttons[2]
        return next_page_button

    def __save_result(self, records):
        csv_file = 'records.csv'

        # Open the CSV file in write mode
        with open(csv_file, 'w', newline='', encoding='utf-8') as file:
            # Define the CSV writer
            csv_writer = csv.DictWriter(file, fieldnames=['name', 'ops', 'image_url'])

            # Write the header row
            csv_writer.writeheader()

            # Write each record as a row
            for record in records:
                csv_writer.writerow(record)

    def __get_ops_type(self, element) -> str:
        """ Click on reveal button and return revealed ops type
        :param element: 'div' with analyzed person
        :return: revealed ops type
        """
        pass

    def __get_interview_link(self, element) -> str:
        """ Get link to interview with analyzed person
        :param element: 'div' with analyzed person
        :return: extracted link to interview
        """
        pass


sp = SubjectivePersonality()
sp.get_persons(1800)
