from requests import get
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json


class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        'User-Agent': 'Mozailla/5.0'
    }
    domain_URL = 'https://www.imdb.com'
    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        self.crawling_threshold = crawling_threshold
        self.not_crawled = []
        self.crawled = []
        self.added_ids = []
        self.url_queue_lock = Lock()
        self.add_list_lock = Lock()

    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        return URL.split('/')[4]

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        with open('IMDB_crawled.json', 'w') as f:
            json.dump(self.crawled, f)

        with open('IMDB_not_crawled.json', 'w') as f:
            json.dump(self.not_crawled, f)

    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        # TODO
        with open('IMDB_crawled.json', 'r') as f:
            self.crawled = json.load(f)

        with open('IMDB_not_crawled.json', 'w') as f:
            self.not_crawled = json.load(f)

        self.added_ids = [movie['id'] for movie in self.crawled]

    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        print(f"getting {URL}")
        res = get(URL, headers=self.headers)
        while res.status_code != 200:
            print(f"retrying for {URL}")
            res = get(URL, headers=self.headers)
        print(f"got {URL}")
        return res

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        # TODO update self.not_crawled and self.added_ids
        response = self.crawl(self.top_250_URL)
        soup = BeautifulSoup(response._content, 'html.parser')
        links = [(self.domain_URL + link.get('href')) for link in soup.find_all('a', class_='ipc-title-link-wrapper') if link.get('href').startswith('/title')]
        self.not_crawled = links

    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.
        TODO: 
            replace WHILE_LOOP_CONSTRAINTS with the proper constraints for the while loop.
            replace NEW_URL with the new URL to crawl.
            replace THERE_IS_NOTHING_TO_CRAWL with the condition to check if there is nothing to crawl.
            delete help variables.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """

        self.extract_top_250()
        futures = []
        crawled_counter = 0

        with ThreadPoolExecutor(max_workers=32) as executor:
            while crawled_counter < self.crawling_threshold:
                self.url_queue_lock.acquire()
                crawled_counter += 1
                URL = self.not_crawled[0]
                self.not_crawled = self.not_crawled[1:]
                self.url_queue_lock.release()
                futures.append(executor.submit(self.crawl_page_info, URL))
                if len(self.not_crawled) == 0:
                    wait(futures)
                    futures = []

    def crawl_page_info(self, URL):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.
        
        Parameters
        ----------
        URL: str
            The URL of the site
        """
        if self.get_id_from_URL(URL) in self.added_ids:
            return
        # print(f"new iteration on {URL}")
        res = self.crawl(URL)
        movie = self.extract_movie_info(res, self.get_imdb_instance(), URL)
        # print(json.dumps(movie))

        self.add_list_lock.acquire()
        print(f"{movie['title']} is Done.")
        self.crawled.append(movie)
        self.added_ids.append(movie['id'])
        self.add_list_lock.release()
        # self.url_queue_lock.acquire()
        for link in movie['related_links']:
            if self.get_id_from_URL(link) not in self.added_ids:
                self.not_crawled.append(link)
        # self.url_queue_lock.release()

    def extract_movie_info(self, res, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        soup = BeautifulSoup(res._content, 'html.parser')

        movie['id'] = self.get_id_from_URL(URL)

        movie['title'] = IMDbCrawler.get_title(soup)
        movie['first_page_summary'] = IMDbCrawler.get_first_page_summary(soup)
        movie['release_year'] = IMDbCrawler.get_release_year(soup)
        movie['mpaa'] = IMDbCrawler.get_mpaa(soup)
        movie['budget'] = IMDbCrawler.get_budget(soup)
        movie['gross_worldwide'] = IMDbCrawler.get_gross_worldwide(soup)
        movie['directors'] = IMDbCrawler.get_director(soup)
        movie['writers'] = IMDbCrawler.get_writers(soup)
        movie['stars'] = IMDbCrawler.get_stars(soup)
        movie['related_links'] = IMDbCrawler.get_related_links(soup)
        movie['genres'] = IMDbCrawler.get_genres(soup)
        movie['languages'] = IMDbCrawler.get_languages(soup)
        movie['countries_of_origin'] = IMDbCrawler.get_countries_of_origin(soup)
        movie['rating'] = IMDbCrawler.get_rating(soup)
        
        summary_soup = BeautifulSoup(self.crawl(IMDbCrawler.get_summary_link(URL))._content, 'html.parser')
        movie['synopsis'] = IMDbCrawler.get_synopsis(summary_soup)
        movie['summaries'] = IMDbCrawler.get_summary(summary_soup)

        reviews_soup = BeautifulSoup(self.crawl(IMDbCrawler.get_review_link(URL))._content, 'html.parser')
        movie['reviews'] = IMDbCrawler.get_reviews_with_scores(reviews_soup)

        return movie

    def get_summary_link(url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            while url[-1] != '/':
                url = url[:-1]
            return (url + "plotsummary")
        except:
            print("failed to get summary link")

    def get_review_link(url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            while url[-1] != '/':
                url = url[:-1]
            return (url + "reviews")
        except:
            print("failed to get review link")

    def get_title(soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
        try:
            return soup.title.string
        except:
            print("failed to get title")

    def get_first_page_summary(soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            return soup.find('span', class_='sc-466bb6c-1').get_text()
        except:
            print("failed to get first page summary for", soup.title.string)
            return "N/A"

    def get_director(soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        try:
            stars = soup.find_all('a', class_='ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link')
            return list(set([star.string for star in stars if star.get('href').endswith('tt_ov_dr')]))
        except:
            print("failed to get director for", soup.title.string)
            return []

    def get_stars(soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            stars = soup.find_all('a', class_='ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link')
            return list(set([star.string for star in stars if star.get('href').endswith('tt_ov_st')]))
        except:
            print("failed to get stars for", soup.title.string)
            return []

    def get_writers(soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
            writers = soup.find_all('a', class_='ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link')
            return list(set([writer.string for writer in writers if writer.get('href').endswith('tt_ov_wr')]))
        except:
            print("failed to get writers for", soup.title.string)
            return []

    def get_related_links(soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        try:
            links = soup.find_all('a', class_='ipc-poster-card__title ipc-poster-card__title--clamp-2 ipc-poster-card__title--clickable')
            return [(IMDbCrawler.domain_URL + link.get('href')) for link in links if link.get('href').startswith('/title')]
        except:
            print("failed to get related links for", soup.title.string)
            return []

    def get_summary(soup):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        try:
            sections = soup.find_all('section', class_='ipc-page-section ipc-page-section--base')
            summaries = sections[0].find_all('div', class_='ipc-html-content-inner-div')
            return [summary.get_text() for summary in summaries]
        except:
            print("failed to get summary for", soup.title.string)
            return []

    def get_synopsis(soup):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        try:
            sections = soup.find_all('section', class_='ipc-page-section ipc-page-section--base')
            summaries = sections[1].find_all('div', class_='ipc-html-content-inner-div')
            return [summary.get_text() for summary in summaries]
        except:
            print("failed to get synopsis for", soup.title.string)
            return []

    def get_reviews_with_scores(soup):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        try:
            reviews = soup.find_all('div', class_='review-container')
            reviews_with_scores = []
            for review in reviews:
                if review.find('span', class_='rating-other-user-rating') is None:
                    continue
                content = review.find('div', class_='text show-more__control')
                score = review.find('span', class_='rating-other-user-rating').find('span').string
                reviews_with_scores.append([content.get_text(), score])
            return reviews_with_scores
        except Exception as err:
            print(f"failed to get reviews for {soup.title.string}:", err)
            return []

    def get_genres(soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        try:
            genres = soup.find_all('a', class_='ipc-chip ipc-chip--on-baseAlt')
            return [genre.string for genre in genres if genre.get('href').endswith('tt_ov_inf')]
        except:
            print("Failed to get generes for", soup.title.string)
            return []


    def get_rating(soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            return soup.find('span', class_='sc-bde20123-1 cMEQkK').string
        except:
            print("failed to get rating for", soup.title.string)
            return "N/A"

    def get_mpaa(soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            candidates = soup.find_all('a', class_='ipc-link ipc-link--baseAlt ipc-link--inherit-color')
            for candidate in candidates:
                if candidate.get('href').endswith('tt_ov_pg'):
                    return candidate.string
            raise Exception("didn't find any")
        except:
            print("failed to get mpaa for", soup.title.string)
            return "N/A"

    def get_release_year(soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            candidates = soup.find_all('a', class_='ipc-link ipc-link--baseAlt ipc-link--inherit-color')
            for candidate in candidates:
                if candidate.get('href').endswith('tt_ov_rdat'):
                    return candidate.string
            raise Exception("didn't find any")
        except:
            print("failed to get release year for", soup.title.string)
            return "N/A"

    def get_languages(soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            candidates = soup.find_all('a', class_='ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link')
            languages = []
            for candidate in candidates:
                if candidate.get('href').endswith('tt_dt_ln'):
                    languages.append(candidate.string)
            return languages
        except:
            print("failed to get languages for", soup.title.string)
            return []

    def get_countries_of_origin(soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            candidates = soup.find_all('a', class_='ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link')
            countries = []
            for candidate in candidates:
                if candidate.get('href').endswith('tt_dt_cn'):
                    countries.append(candidate.string)
            return countries
        except:
            print("failed to get countries of origin for", soup.title.string)
            return []

    def get_budget(soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            candidates = soup.find_all('span', class_='ipc-metadata-list-item__label')
            for candidate in candidates:
                if candidate.string == 'Budget':
                    return candidate.findNext('span').string
            raise Exception("didn't find any")
        except:
            print("failed to get budget for", soup.title.string)
            return "N/A"

    def get_gross_worldwide(soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        try:
            candidates = soup.find_all('span', class_='ipc-metadata-list-item__label')
            for candidate in candidates:
                if candidate.string == 'Gross worldwide':
                    return candidate.findNext('span').string
            raise Exception("didn't find any")
        except:
            print("failed to get gross worldwide for", soup.title.string)
            return "N/A"


def main():
    print("started!")
    imdb_crawler = IMDbCrawler(crawling_threshold=5000)
    print("going!")
    # imdb_crawler.crawl_page_info('https://www.imdb.com/title/tt0786945/')
    # imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    print("writing to json ...")
    print(len(imdb_crawler.crawled))
    imdb_crawler.write_to_file_as_json()


if __name__ == '__main__':
    main()
