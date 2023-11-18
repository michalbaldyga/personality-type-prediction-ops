from backend.gathering_data.YoutubeBrowser import YoutubeBrowser

browser = YoutubeBrowser()
CSV_DIR = '../../static/csv/'


def get_interviews(records_file_path):
    results = [["name", "ops", "interview_link"]]

    with open(records_file_path) as record_file:
        lines = record_file.readlines()[1:]

        for line in lines:
            cols = line.split(';') if ';' in line else line.split(',')
            name = cols[0]
            is_member = int(cols[4])

            yt_link = cols[5].replace('\n', '')

            interview_link = get_interview_link(name, yt_link, is_member)

            if interview_link is not None:
                results.append([name, cols[1], interview_link])

    with open(f"{CSV_DIR}interview_links.csv", "w", newline='') as output_file:
        output_file.writelines(f"{';'.join(line)}\n" for line in results[:-1])
        output_file.write(';'.join(results[-1]))


def get_interview_link(name, link, is_member):
    def __validate_name(person_name):
        parts = person_name.split()

        if len(parts) < 2:
            return False
        if len(parts) == 2 and len(parts[1]) < 2:
            return False
        return True

    if link is None or len(link) == 0:
        if __validate_name(name):
            path = 'https://www.youtube.com/results?search_query='
            link = f"{path}{name.replace(' ', '%20')}%20interview"
        else:
            return None
    if 'search_query' in link:
        if __validate_name(name) and not is_member:
            return browser.get_interview_by_search_link(link, name)
    else:
        return link

    return None


if __name__ == '__main__':
    get_interviews(f"{CSV_DIR}records.csv")
