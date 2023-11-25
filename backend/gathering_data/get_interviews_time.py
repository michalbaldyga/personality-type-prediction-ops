import re
import requests


def sum_interviews_time():
    result = 0.0
    idx = 1

    with open("../../static/csv/interview_links.csv") as file:
        lines = file.readlines()

        for line in lines[1:]:
            url = line.split(';')[2].replace('\n', '')

            page = requests.get(url)
            content = page.content.decode()

            try:
                duration = get_video_duration(content)
                if duration == 0.0:
                    raise Exception(f"{idx}: Error {url} .Video duration equals 0")
                result += duration
                print(f"{idx}: {url} - {result}")
                idx += 1
            except:
                print(f"{idx}: {url} - Error occured")
                idx += 1

    return result


def get_video_duration(content):
    duration_tag = re.findall('meta itemprop="duration" content="PT[0-9A-Z]+', content)
    duration = duration_tag[0].split('"')[-1]
    return __convert_duration_to_minutes(duration)


def __convert_duration_to_minutes(duration):
    hours_match = re.search(r'(\d+)H', duration)
    minutes_match = re.search(r'(\d+)M', duration)
    seconds_match = re.search(r'(\d+)M', duration)

    # Initialize variables for each component (hours, minutes)
    hours = int(hours_match.group(1)) if hours_match else 0
    minutes = int(minutes_match.group(1)) if minutes_match else 0
    seconds = int(seconds_match.group(1)) if seconds_match else 0

    # Calculate the total duration in minutes
    total_minutes = float((hours * 60) + minutes + seconds / 60.0)
    return total_minutes


if __name__ == "__main__":
    print(f"Total time: {sum_interviews_time()}")
