# streamlit run UI/main.py

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(sys.path[0], '../..'))
sys.path.append(os.path.abspath('.'))
# sys.path.append("E:\University\Term 6\Modern Information Retreival\Project\MIR-imdb")
from Logic import utils
import time
from enum import Enum
import random
from Logic.core.utility.snippet import Snippet
from Logic.core.link_analysis.analyzer import LinkAnalyzer
from Logic.core.indexer.index_reader import Index_reader, Indexes

snippet_obj = Snippet()


class color(Enum):
    RED = "#FF0000"
    GREEN = "#00FF00"
    BLUE = "#0000FF"
    CYAN = "#00FFFF"
    MAGENTA = "#FF00FF"


STAR_LINK = "https://i.ibb.co/CMN1wTh/star.png"

############################################## Streamlit functions

def fill_bar(data_value, max_value, bar_width=480, bar_height=25, fill_color="#ff4b4b"):
    """
    This function creates a progress bar filled based on data and max values.

    Args:
        data_value: The current value to represent as fill.
        max_value: The maximum value for the bar.
        bar_width: Optional width of the bar (default 100).
        bar_height: Optional height of the bar (default 20).
        fill_color: Optional color for the filled portion (default blue).

    Returns:
        None
    """
    # Calculate fill percentage
    fill_pct = min(100, int(data_value / max_value * 100))

    # Create filled portion
    fill_width = int(bar_width * fill_pct / 100)
    st.write(" " * int((bar_width - fill_width) / 2), unsafe_allow_html=True)

    # Create the bar using a single markdown call with combined styling
    combined_style = f"""
    <div style='background-color: grey; width: {bar_width}px; height: {bar_height}px; border-radius: 3x;'>
        <div style='background-color: {fill_color}; width: {fill_width}px; height: {bar_height}px; white-space: nowrap; overflow: visible; border-radius: 3x;'>
            &nbsp Relevancy score: {data_value}
        </div>
    </div>
    """
    fill_width = int(bar_width * fill_pct / 100)
    st.markdown(combined_style.format(width=fill_width), unsafe_allow_html=True)


##############################################



def get_top_x_movies_by_rank(x: int, results: list):
    path = "index/"  # Link to the index folder
    document_index = Index_reader(path, Indexes.DOCUMENTS)
    corpus = []
    root_set = []
    for movie_id, movie_detail in document_index.index.items():
        movie_title = movie_detail["title"]
        stars = movie_detail["stars"]
        corpus.append({"id": movie_id, "title": movie_title, "stars": stars})

    for element in results:
        movie_id = element[0]
        movie_detail = document_index.index[movie_id]
        movie_title = movie_detail["title"]
        stars = movie_detail["stars"]
        root_set.append({"id": movie_id, "title": movie_title, "stars": stars})
    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=x)
    return actors, movies


def get_summary_with_snippet(movie_info, query):
    summary = movie_info['first_page_summary']
    snippet, not_exist_words = snippet_obj.find_snippet(summary, query)
    if "***" in snippet:
        snippet = snippet.split()
        for i in range(len(snippet)):
            current_word = snippet[i]
            if current_word.startswith("***") and current_word.endswith("***"):
                current_word_without_star = current_word[3:-3]
                snippet[i] = f"<b><font size='4' color={random.choice(list(color)).value}>{current_word_without_star}</font></b>"
        return ' '.join(snippet)
    return summary

    summary = movie_info["first_page_summary"]
    snippet, not_exist_words = snippet_obj.find_snippet(summary, query)
    if "***" in snippet:
        snippet = snippet.split()
        for i in range(len(snippet)):
            current_word = snippet[i]
            if current_word.startswith("***") and current_word.endswith("***"):
                current_word_without_star = current_word[3:-3]
                summary = summary.lower().replace(
                    current_word_without_star,
                    f"<b><font size='4' color={random.choice(list(color)).value}>{current_word_without_star}</font></b>",
                )
    return summary


def search_time(start, end):
    st.success("Search took: {:.6f} milli-seconds".format((end - start) * 1e3))


def search_handling(
    search_button,
    search_term,
    search_max_num,
    search_weights,
    search_method,
    unigram_smoothing,
    alpha,
    lamda,
    filter_button,
    num_filter_results,
):
    if filter_button:
        if "search_results" in st.session_state:
            top_actors, top_movies = get_top_x_movies_by_rank(
                num_filter_results, st.session_state["search_results"]
            )
            st.header(f"**Top {num_filter_results} Actors:**")
            for idx, actor in enumerate(top_actors):
                st.markdown(
                    f"""<span>
                    <h5 style='color:{list(color)[idx % len(list(color))].value}'>
                    {idx+1}- <a href='https://www.imdb.com/find/?q={actor}', style='color:{list(color)[idx % len(list(color))].value}'>
                    {actor}
                    </a>
                    </h5>
                    </span>""",
                    unsafe_allow_html=True,
                )
            st.divider()

        st.header(f"**Top {num_filter_results} Movies:**")
        for i in range(len(top_movies)):
            card = st.columns([14, 6])
            info = utils.get_movie_by_id(top_movies[i], utils.movies_dataset)
            with card[0].container():
                st.title(f"[{info['title']}]({info['URL']})")
                st.markdown(
                    f"<b><font size = '4'>Summary:</font></b> {get_summary_with_snippet(info, search_term)}",
                    unsafe_allow_html=True,
                )

                with st.container():
                    st.markdown("**Directors:**")
                    num_authors = len(info["directors"])
                    authors = "".join(author + ", " for author in info["directors"])
                    st.text(authors[:-2])

                with st.container():
                    st.markdown("**Stars:**")
                    num_authors = len(info["stars"])
                    stars = "".join(star + ", " for star in info["stars"])
                    st.text(stars[:-2])

            with st.container():
                st.write("**Genres:**")
                tmp_card = st.columns(2)
                with tmp_card[0].container():
                    num_topics = len(info["genres"])
                    topic_card = st.columns(num_topics)
                    for j in range(num_topics):
                        with topic_card[j].container():
                            st.link_button(info['genres'][j], f"https://www.imdb.com/search/title/?genres={info['genres'][j]}", type='primary')
            
            with card[1].container():
                st.image(info["Image_URL"], use_column_width=True)
                star_card = st.columns(5)
                print(float((info["rating"])))
                star_count = int(float((info["rating"]))) // 2
                for j in range(star_count):
                    with star_card[j].container():
                        st.image(STAR_LINK, use_column_width=True)
                
                tmp_card = st.columns(3)
                with tmp_card[1].container():
                    st.button(info["rating"], disabled=True, help="It's the rating you idiot! What else did you think it would be?!", key=f"{i}")
  
            st.divider()
        return

    if search_button:
        corrected_query = utils.correct_text(search_term, utils.documents)

        if corrected_query != search_term:
            st.warning(f"Your search terms were corrected to: {corrected_query}")
            search_term = corrected_query

        with st.spinner("Searching..."):
            time.sleep(0.5)  # for showing the spinner! (can be removed)
            start_time = time.time()
            result = utils.search(
                search_term,
                search_max_num,
                search_method,
                search_weights,
                unigram_smoothing=unigram_smoothing,
                alpha=alpha,
                lamda=lamda,
            )
            if "search_results" in st.session_state:
                st.session_state["search_results"] = result
            print(f"Result: {result}")
            end_time = time.time()
            if len(result) == 0:
                st.warning("No results found!")
                return

            search_time(start_time, end_time)

        max_relevane = max([x[1] for x in result])
        for i in range(len(result)):
            card = st.columns([14, 6])
            info = utils.get_movie_by_id(result[i][0], utils.movies_dataset)
            with card[0].container():
                st.title(f"[{info['title']}]({info['URL']})")
                fill_bar(result[i][1], max_relevane)

                st.markdown(
                    f"<b><font size = '4'>Summary:</font></b> {get_summary_with_snippet(info, search_term)}",
                    unsafe_allow_html=True,
                )
                with st.container():
                    st.markdown("**Directors:**")
                    num_authors = len(info["directors"])
                    authors = "".join(author + ", " for author in info["directors"])
                    st.text(authors[:-2])

                with st.container():
                    st.markdown("**Stars:**")
                    num_authors = len(info["stars"])
                    stars = "".join(star + ", " for star in info["stars"])
                    st.text(stars[:-2])

            with st.container():
                st.write("**Genres:**")
                tmp_card = st.columns(2)
                with tmp_card[0].container():
                    num_topics = len(info["genres"])
                    topic_card = st.columns(num_topics)
                    for j in range(num_topics):
                        with topic_card[j].container():
                            st.link_button(info['genres'][j], f"https://www.imdb.com/search/title/?genres={info['genres'][j]}", type='primary')
            
            with card[1].container():
                st.image(info["Image_URL"], use_column_width=True)
                star_card = st.columns(5)
                print(float((info["rating"])))
                star_count = int(float((info["rating"]))) // 2
                for j in range(star_count):
                    with star_card[j].container():
                        st.image(STAR_LINK, use_column_width=True)
                
                tmp_card = st.columns(3)
                with tmp_card[1].container():
                    st.button(info["rating"], disabled=True, help="It's the rating you idiot! What else did you think it would be?!", key=f"{i}")

            st.divider()

        st.session_state["search_results"] = result
        if "filter_state" in st.session_state:
            st.session_state["filter_state"] = (
                "search_results" in st.session_state
                and len(st.session_state["search_results"]) > 0
            )


def main():
    # page_bg_img = '''
    #     <style>
    #     [data-testid="stAppViewContainer"] {
    #     background-image: url("https://i.ibb.co/vzD6q3n/bg.png");
    #     background-size: cover;
    #     }
    #     </style>
    # '''
    page_bg_img = '''
        <style>
        [data-testid="stAppViewContainer"] {
        background-image: url("https://i.ibb.co/6t6rGXQ/bg.png");
        background-size: cover;
        }
        </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.logo("https://i.ibb.co/rcdG6gm/rottenpotatoes.webp")

    colorline = '''
        <style>
        hr {
            border-color: tomato;
        }
        </style>
    '''
    st.markdown(colorline, unsafe_allow_html=True)
    
    title_col = st.columns([9, 11])
    with title_col[0].container():
        st.title("Rotten Potatoes")
    with title_col[1].container():
        st.image("https://i.ibb.co/rcdG6gm/rottenpotatoes.webp", width=85)
    st.write(
        "This is a simple search engine for IMDB movies. You can search through IMDB dataset and find the most relevant movie to your search terms. But it can only show you Shawshank poster, So Yeah ..."
    )
    st.markdown(
        '<span style="color:red">Developed By: MIR Team at Sharif University (And the tears of a poor student!)</span>',
        unsafe_allow_html=True,
    )

    search_term = st.text_input("**Search Term**")
    with st.expander("**Advanced Search**"):
        search_max_num = st.number_input(
            "Maximum number of results", min_value=5, max_value=100, value=10, step=5
        )
        weight_stars = st.slider(
            "Weight of stars in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )

        weight_genres = st.slider(
            "Weight of genres in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )

        weight_summary = st.slider(
            "Weight of summary in search",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.1,
        )
        slider_ = st.slider("Select the number of top movies to show", 1, 10, 5)

        search_weights = [weight_stars, weight_genres, weight_summary]
        search_method = st.selectbox(
            "Search method", ("ltn.lnn", "ltc.lnc", "OkapiBM25", "unigram")
        )

        unigram_smoothing = None
        alpha, lamda = None, None
        if search_method == "unigram":
            unigram_smoothing = st.selectbox(
                "Smoothing method",
                ("naive", "bayes", "mixture"),
            )
            if unigram_smoothing == "bayes":
                alpha = st.slider(
                    "Alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )
            if unigram_smoothing == "mixture":
                alpha = st.slider(
                    "Alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )
                lamda = st.slider(
                    "Lambda",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                )

    if "search_results" not in st.session_state:
        st.session_state["search_results"] = []

    tmp_col = st.columns([2, 13])
    with tmp_col[0].container():    
        search_button = st.button("Search!")
    with tmp_col[1].container():
        filter_button = st.button("Filter movies by ranking")

    search_handling(
        search_button,
        search_term,
        search_max_num,
        search_weights,
        search_method,
        unigram_smoothing,
        alpha,
        lamda,
        filter_button,
        slider_,
    )


if __name__ == "__main__":
    main()
