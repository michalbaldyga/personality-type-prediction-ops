""" Class used to get persons from subjectivepersonality.com with ops type and link to interview """

from selenium import webdriver
from selenium.webdriver.common.by import By

ELEMENTS_ON_PAGE = 50


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

        while len(results) < limit:
            page_limit = min(ELEMENTS_ON_PAGE, limit - len(results))
            elements = self.__get_elements_from_page(page_limit)

            for element in elements:
                ops_type = self.__get_ops_type(element)
                link = self.__get_interview_link(element)
                results.append((ops_type, link))

            self.__go_to_next_page()

        return results

    def __get_elements_from_page(self, limit) -> list:
        """ Get all elements with person from current page
        :param limit: maximum number of elements to get
        :return: list of elements
        """
        pass

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
