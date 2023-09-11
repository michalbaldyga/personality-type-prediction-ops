from selenium import webdriver


class SubjectivePersonality:
    def __init__(self):
        self.driver = webdriver.Chrome()
        self.driver.get("https://app.subjectivepersonality.com/ops/class")
