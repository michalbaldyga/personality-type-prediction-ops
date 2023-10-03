""" Class used to get persons from subjectivepersonality.com with ops type and link to interview """
from typing import Tuple

from selenium import webdriver
from selenium.webdriver.common.by import By

ELEMENTS_ON_PAGE = 50


def get_image_url(text: str) -> str:
    tag = '&quot;'
    start = text.find(tag)
    end = text.find(tag, start+1)
    result = text[start+len(tag):end]
    return result


def get_name_and_ops(text: str) -> Tuple[str, str]:
    result = text.split('\n')
    return result[0], result[1]


class SubjectivePersonality:
    def __init__(self):
        """ Initialize selenium driver """
        self.driver = webdriver.Chrome()
        self.driver.get("https://app.subjectivepersonality.com/ops/class")

    def get_persons(self, limit: int) -> list:
        """ Get persons from page with type and link to interview
        :param limit: maximum number of persons to get
        :return: list of persons: [(ops type, link to interview)]
        """
        results = []
        show_types_button = self.driver.find_element(By.CLASS_NAME, 'btn-sm')
        show_types_button.click()

        while len(results) < limit:
            elements_with_text = self.__get_elements_from_page('type-record')
            elements_with_images = self.__get_elements_from_page('image-img')

            # we need to skip every second element
            elements_with_images = elements_with_images[::2]

            for text, image in zip(elements_with_text, elements_with_images):
                image_url = get_image_url(image.get_attribute("outerHTML"))
                name, ops = get_name_and_ops(text.text)
                results.append({'name': name, 'ops': ops, 'image_url': image_url})

            self.__go_to_next_page()

        return results

    def __get_elements_from_page(self, class_name: str) -> list:
        """ Get class elements from current page
        :param class_name: class name
        :return: list of elements
        """
        return self.driver.find_elements(By.CLASS_NAME, class_name)

    def __go_to_next_page(self):
        """ Click "Next" button to go to next page
        :return: None
        """
        pass

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
sp.get_persons(1900)
