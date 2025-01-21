import math
from itertools import permutations, combinations
from typing import Union, Literal

import numpy as np
import pandas as pd
from transformers import pipeline
import tomotopy
from tomotopy import LDAModel
from tqdm import tqdm
import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from wordcloud import WordCloud


class LDAEngine:
    """ Class that wraps a tomotopy.LDAModel with additional evaluation and interpretation functions. """
    def __init__(self,
                 text_frame: Union[pd.DataFrame, str] = None,
                 text_column = "Abstract",
                 time_column = "PublicationDate",
                 id_column='PMID',
                 model_file:str = None,
                 topic_labels: [str] = None,
                 n_words_per_topic: int = 8
                 ):
        """
        Initialize an LDAEngine instance.

        This method sets up the LDAEngine with the provided text data and parameters, either loading a model
        or preparing for a new one to be trained.

        Parameters
        ----------
        text_frame : Union[pd.DataFrame, str], optional
            DataFrame or JSON file path containing text data.
        text_column : str, optional
            Name of the column containing text data. Default is "Abstract".
        time_column : str, optional
            Name of the column containing publication dates. Default is "PublicationDate".
        id_column : str, optional
            Name of the column containing unique IDs for texts. Default is 'PMID'.
        model_file : str, optional
            Path to a pre-trained LDAModel file to load. Default is None.
        topic_labels : list of str, optional
            Pre-defined labels for topics. Default is None.
        n_words_per_topic : int, optional
            Number of words to display per topic. Default is 8.

        Examples
        --------
        >>> engine = LDAEngine(text_frame="data.json", topic_labels=["Topic A", "Topic B"])
        """
        self.text_frame = text_frame if isinstance(text_frame, pd.DataFrame) else pd.read_json(text_frame)
        self.text_column = text_column  # text column name
        self.time_column = time_column  # publication date column name
        self.id_column = id_column  # text id column
        self.model = None if model_file is None else LDAModel.load(str(model_file))  # eventually load model from .bin file
        self.n_words_per_topic = n_words_per_topic  # number of words per topic to consider and display
        self.topic_labels = None if topic_labels is None else list(topic_labels)  # eventually pre-defined topic labels

    def __repr__(self):
        """ String representation """
        if self.model is not None: self.model.summary()
        return f"LDAEngine instance with {len(self.text_frame)} articles."

    def __str__(self):
        """ Print operator """
        return self.__repr__()

    @property
    def text_series(self) -> pd.Series:
        """
        Get the text entries from the text_frame.

        Returns
        -------
        pd.Series
            Series containing text data
        """
        return self.text_frame.loc[:, self.text_column]

    @text_series.setter
    def text_series(self, value):
        """
        Set text entries in the text_frame.

        Parameters
        ----------
        value : pd.Series
            New text data to replace the current text column.
        """
        self.text_frame.loc[:, self.text_column] = value

    def initialise_model(self, n_topics: int, **model_kwargs):
        """
        Initialize a tomotopy.LDAModel instance and add documents.

        This function prepares the model with the specified number of topics and adds
        the text data from the text_frame to the model.

        Parameters
        ----------
        n_topics : int
            Number of topics for the model.
        **model_kwargs : dict
            Additional parameters for the LDAModel initialization.
        """
        self.model = LDAModel(k=n_topics, **model_kwargs)  # initialise model
        _ = [self.model.add_doc(words=text.split()) for text in self.text_series]  # add docs

    def evaluate_umass_coherence(self, n_topics_range=(10, 50, 5), verbose=False, iterations=500) -> pd.Series:
        """
        Evaluate and plot u-mass coherence across different topic numbers.

        This method returns a Series of coherence scores for the specified range of topics.

        Parameters
        ----------
        n_topics_range : tuple of int, optional
            Range of topic numbers as (start, stop, step). Default is (10, 50, 5).
        verbose : bool, optional
            If True, prints additional information. Default is False.
        iterations : int, optional
            Number of iterations for each model. Default is 500.

        Returns
        -------
        pd.Series
            Coherence scores for each topic number.
        """
        return plot_umass_scores(self.text_series, k_range=n_topics_range, verbose=verbose, iterations=iterations)

    def train(self, iterations=1000, show_progress=True, include_training_log=True, **train_kwargs):
        """
        Train the LDAModel instance.

        This method trains the model for the specified number of iterations, optionally saving a training log
        and plotting the convergence of the log-likelihood.

        Parameters
        ----------
        iterations : int, optional
            Number of training iterations. Default is 1000.
        show_progress : bool, optional
            If True, displays progress during training. Default is True.
        include_training_log : bool, optional
            If True, saves and plots the training log. Default is True.
        **train_kwargs : dict
            Additional training parameters for the LDAModel.
        """
        training_log = TrainingLog() if include_training_log else None  # initialise log file
        self.model.train(iter=iterations, show_progress=show_progress, callback=training_log,
                         callback_interval=iterations//100, **train_kwargs)  # callback interval defines logging freq.
        if include_training_log: training_log.plot()  # plot convergence

    @property
    def n_topics(self) -> int:
        """
        Get the number of topics in the model.

        Returns
        -------
        int
            Number of topics in the model.

        Raises
        ------
        AttributeError
            If the model has not been initialized.
        """
        if self.model is None: raise AttributeError("Model needs to be initialised first!")
        else: return self.model.k

    @property
    def word_frequencies(self) -> pd.DataFrame:
        """
        Get word frequencies in the model.

        Returns
        -------
        pd.DataFrame
            DataFrame containing word frequencies.
        """
        return get_word_frequency(self.model)

    @property
    def words_per_topic(self) -> pd.DataFrame:
        """
        Get distinctive words for each topic.

        Returns
        -------
        pd.DataFrame
            DataFrame containing distinctive words for each topic.
        """
        word_frame = get_words_per_topic(self.model, no_of_words=self.n_words_per_topic)
        if self.topic_labels is not None:
            word_frame['Topic Label'] = self.topic_labels  # eventually include labels
            word_frame.set_index('Topic Label', inplace=True)  # and set as index:
        return word_frame

    @property
    def tokenization_dict(self) -> dict:
        """
        Get the word-to-token mapping.

        Returns
        -------
        dict
            Dictionary mapping words to their token values.
        """
        return get_tokenization(self.model)

    @property
    def publication_years(self) -> pd.Series:
        """
        Get the publication years from the text_frame.

        Returns
        -------
        pd.Series
            Series containing publication years.
        """
        return get_publication_year(self.text_frame, time_column=self.text_column)

    def get_time_sliced_text_frame(self, start_year: int = None, end_year: int = None, verbose=True) -> pd.DataFrame:
        """
        Get a subset of texts published within a specified time interval.

        Parameters
        ----------
        start_year : int, optional
            Start of the time interval. Default is None.
        end_year : int, optional
            End of the time interval. Default is None.
        verbose : bool, optional
            If True, prints additional information. Default is True.

        Returns
        -------
        pd.DataFrame
            Subset of the text_frame within the specified time interval.
        """
        return get_slice_publication_year(self.text_frame, start_year=start_year, end_year=end_year, verbose=verbose,
                                          time_column=self.time_column)

    ######################## Topic Labelling Functions ########################
    def label_topics_with_llm(self, model_name:str = "google/flan-t5-large", long_prompt=True, verbose=True) -> None:
        """
        Utilize a Hugging Face text-generation pipeline to label topics.

        This function leverages a large language model (LLM) to generate labels
        for the topics extracted by the underlying model. The generated labels
        are stored in the `topic_labels` attribute for further use.

        Parameters
        ----------
        model_name : str, optional
            The name of the Hugging Face model to be used for text generation,
            by default "google/flan-t5-large".
        long_prompt : bool, optional
            Whether to use a longer prompt for better results, by default True.
        verbose : bool, optional
            Whether to print additional information during execution, by default True.

        Returns
        -------
        None

        See Also
        --------
        label_topics_llm : Internal function to label topics using an LLM.
        label_validation_prompt : Generate validation prompts for manual review.
        """
        # infer and save topic labels:
        generator = pipeline("text2text-generation", model=model_name)  # initialise llm model
        labelled_topic_word_frame = label_topics_llm(self.model, generator, long_prompt=long_prompt, verbose=verbose)
        self.topic_labels = list(labelled_topic_word_frame.index)

        # validation prompt:
        print(f"Topic labels have been generated with hugging-face's {model_name}. It is recommended to evaluate those manually!")
        print(f"Additionally, consider running this validation prompt through another LLM:\n\n{label_validation_prompt(labelled_topic_word_frame)}")

    def export_results(self, output_path: str, title_prefix: str = "") -> None:
        """
        Export the trained model and topic-related data to disk.

        This function saves the trained model as a binary file and exports the
        labeled topics and their associated words to a CSV file, if available.

        Parameters
        ----------
        output_path : str
            Path to the directory where the output files will be saved.
        title_prefix : str, optional
            A prefix to include in the file names, by default an empty string.

        Returns
        -------
        None
        """
        model_save_title = str(output_path) + "/" + str(title_prefix) + f"Trained LDAModel {self.n_topics} Topics {len(self.text_frame)} Articles.bin"
        self.model.save(model_save_title)
        if self.topic_labels is not None:
            topic_save_title = str(output_path) + "/" + str(title_prefix) + f"Topic-word-Dist {self.n_topics} Topics {len(self.text_frame)} Articles.csv"
            self.words_per_topic.to_csv(topic_save_title)

    def get_topics_per_article(self, article_index: int, verbose=True, n_topics=5):
        """
        Analyze the topic distribution of a single article.

        This function retrieves the top topics associated with a specific article
        based on its index in the corpus.

        Parameters
        ----------
        article_index : int
            Index of the article to analyze.
        verbose : bool, optional
            Whether to print the article content and associated topics, by default True.
        n_topics : int, optional
            Number of top topics to retrieve, by default 5.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the topics and their respective contributions.
        """
        # infer topics:
        topics = get_topics_per_document(self.model, article_index, topics_to_include=n_topics, topic_name_list=self.topic_labels)

        # eventually print article and topic_words:
        if verbose:
            print("Article to be analysed:\n", self.text_series.loc[article_index])
            for index, row in topics.iterrows():
                print(f"{index}: '{row['Topic Name']}' with words {list(self.words_per_topic.iloc[int(row['Topic Index']), :])[:-1]}")

        return topics

    @property
    def topic_occurrences(self) -> pd.DataFrame:
        """
        Get a list of topics and their occurrences in the corpus.

        This property returns a DataFrame with topics sorted in descending order
        by their frequency of occurrence in the corpus.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with columns for topic names, indexes, and occurrences.
        """
        occurrence_frame = count_topic_occurrences(self.model, occurrence_threshold_topics=1, topic_name_list=self.topic_labels)
        occurrence_frame.sort_values(by='Occurrences', ascending=False, inplace=True)
        return occurrence_frame

    def get_labelled_text_frame(self, title_column='Title', author_column='Authors', journal_column='Journal') -> pd.DataFrame:
        """
        Create a labeled text frame with topic assignments and metadata.

        This function constructs a DataFrame containing the text, topics, and
        additional metadata such as publication year, title, authors, and journal.

        Parameters
        ----------
        title_column : str, optional
            Name of the column containing article titles, by default 'Title'.
        author_column : str, optional
            Name of the column containing author names, by default 'Authors'.
        journal_column : str, optional
            Name of the column containing journal names, by default 'Journal'.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with text data, topic assignments, and metadata.
        """
        # columns to be included:
        column_list = [self.id_column, self.text_column, self.time_column]
        if author_column is not None: column_list.append(author_column)
        if journal_column is not None: column_list.append(journal_column)

        # copy relevant columns:
        relevant_output_table = self.text_frame.loc[:, column_list].copy()
        relevant_output_table['PublicationYear'] = self.publication_years
        relevant_output_table['Topic Number'], relevant_output_table['Topic Label'] = get_topic_assignments_for_corpus(self.model,
                                                                             self.text_series, self.topic_labels)
        return relevant_output_table

    ######################## Plotting Functions ########################
    def plot_word_clouds(self, cloud_word_count: int = 50,
                         topics_to_evaluate: [int] = None) -> None:
        """
        Plot word clouds for selected topics.

        This function generates word clouds for the specified topics, using the
        most frequent words associated with each topic. If topic labels are
        available, they are included in the visualization.

        Parameters
        ----------
        cloud_word_count : int, optional
            The maximum number of words to include in each word cloud, by default 50.
        topics_to_evaluate : list of int, optional
            A list of topic indices to evaluate. If not provided, the default is [0, 1, 2].

        Returns
        -------
        None
        """
        # include topic labels if already labelled
        topics_to_evaluate = [0, 1, 2] if topics_to_evaluate is None else topics_to_evaluate
        topic_labels = [self.topic_labels[topic_ind] for topic_ind in topics_to_evaluate] if self.topic_labels is not None else None
        # plot:
        plot_word_clouds(self.model, cloud_word_count=cloud_word_count, topics_to_evaluate=topics_to_evaluate,
                         topic_labels=topic_labels)

    def plot_publications_per_year(self, time_interval: tuple = (None, None), years_to_summarize=1) -> None:
        """
        Plot a histogram of publications per year.

        This function creates a histogram showing the number of publications
        per year, optionally summarizing over a specified number of years and
        within a specific time interval.

        Parameters
        ----------
        time_interval : tuple, optional
            A tuple specifying the start and end years for the plot. Use None
            for either value to include all available years, by default (None, None).
        years_to_summarize : int, optional
            The number of years to group together for summarization, by default 1.

        Returns
        -------
        None
        """
        plot_publications_per_year(self.text_frame, self.time_column, time_interval[0], time_interval[1], years_to_summarize=years_to_summarize)

    def plot_topic_graph(self, graph=None, positions=None,
                         save_dir: str = None, save_title_prefix: str = "",
                         time_slice: tuple = (None, None),
                         community_algorithm:Literal[None, 'greedy', 'leiden', 'multilevel'] = None,
                         occurrence_threshold_topics=1, co_occurrence_threshold_topics=2,
                         mpl_colorpalette="tab20c", node_scale_factor=30, edge_scale_factor=None, edge_power=1.4,
                         optimal_node_distance=2.1,
                         **plot_kwargs):
        """
        Plot a topic network diagram.

        This function creates a network diagram to visualize relationships
        between topics. It supports community detection and filtering of articles
        based on time slices. The graph can be saved as an image file if a save
        directory is specified.

        Parameters
        ----------
        graph : networkx.Graph, optional
            A pre-initialized graph object. If not provided, one is created based
            on the model, by default None.
        positions : dict, optional
            Node positions for the graph layout. If not provided, positions are
            generated automatically, by default None.
        save_dir : str, optional
            Directory to save the graph image, by default None.
        save_title_prefix : str, optional
            Prefix for the saved file name, by default an empty string.
        time_slice : tuple, optional
            A tuple specifying the start and end years for filtering articles,
            by default (None, None).
        community_algorithm : {'greedy', 'leiden', 'multilevel'}, optional
            The algorithm used to detect communities in the graph. If None, no
            community detection is performed, by default None.
        occurrence_threshold_topics : int, optional
            Minimum number of occurrences for a topic to be included, by default 1.
        co_occurrence_threshold_topics : int, optional
            Minimum number of co-occurrences between topics for an edge to be included,
            by default 2.
        mpl_colorpalette : str, optional
            Matplotlib color palette to use for node and edge coloring, by default "tab20c".
        node_scale_factor : int, optional
            Scaling factor for node sizes, by default 30.
        edge_scale_factor : int, optional
            Scaling factor for edge widths. If None, no scaling is applied, by default None.
        edge_power : float, optional
            Power to which edge weights are raised for scaling, by default 1.4.
        optimal_node_distance : float, optional
            Desired average distance between nodes, by default 2.1.
        **plot_kwargs : dict
            Additional keyword arguments for the plotting function.

        Returns
        -------
        tuple
            A tuple containing the graph object and the positions dictionary.
        """
        # eventually select article subset:
        article_subset = self.get_time_sliced_text_frame(start_year=time_slice[0], end_year=time_slice[1]) if time_slice != (None, None) else None
        if article_subset is not None and graph is not None:
            print("Time slice will not be considered because pre-defined graph was also provided to function.")

        # if not provided, initialise graph and position mapping:
        if graph is None or positions is None:
            graph, positions = initialise_topic_graph(self.model, topic_labels = self.topic_labels,
                                                      article_subset=article_subset, hidden=True,
                                                      occurrence_threshold_topics=occurrence_threshold_topics,
                                                      co_occurrence_threshold_topics=co_occurrence_threshold_topics,
                                                      mpl_colorpalette=mpl_colorpalette, node_scale_factor=node_scale_factor,
                                                      edge_scale_factor=edge_scale_factor, edge_power=edge_power,
                                                      optimal_node_distance=optimal_node_distance)

        # calculate communities if algorithm was defined (else set respective arguments to None):
        if community_algorithm is not None:
            community_memberships = get_communities(graph, algorithm=community_algorithm)
            edge_color_dict, node_color_dict = get_community_edge_node_colors(graph, community_memberships,
                                                                              mpl_colormap="rainbow")
        else:
            edge_color_dict = node_color_dict = community_memberships = None

        # construct save_title if save_dir was provided (else set respective argument to None):
        if save_dir is not None:
            if community_algorithm is not None: save_title_prefix += "With Communities "
            if time_slice[0] is not None: save_title_prefix += f"From {time_slice[0]} "
            if time_slice[1] is not None: save_title_prefix += f"Til {time_slice[1]} "
            save_title = str(save_dir) + "/" + save_title_prefix + f"Topic network {len(self.text_frame)} Articles {self.n_topics} Topics.png"
        else:
            if save_title_prefix is not None: print("Save title prefix will not be considered, because save_dir not provided.")
            save_title = None

        # plot graph:
        plot_topic_graph(graph, positions, topic_labels=self.topic_labels,
                         node_colors=node_color_dict, edge_colors=edge_color_dict,
                         community_memberships=community_memberships,
                         save_title=save_title, **plot_kwargs)

        return graph, positions


######################## Auxiliary Functions ########################
def get_tokenization(model: LDAModel) -> dict:
    """
    Generate a tokenization mapping for the vocabulary used by the LDA model.

    This function creates a dictionary mapping words in the model's vocabulary
    to their corresponding indices.

    Parameters
    ----------
    model : LDAModel
        The trained LDA model containing the vocabulary.

    Returns
    -------
    dict
        A dictionary where keys are words from the vocabulary and values
        are their corresponding indices.
    """
    return {model.used_vocabs[ind]: ind for ind in range(len(model.used_vocabs))}


def get_publication_year(input_frame: pd.DataFrame, time_column='PublicationDate') -> pd.Series:
    """
    Extract the publication year from a date column in a DataFrame.

    This function processes a time column following the ISO8601 format
    (e.g., "2020-01-01") and returns a series of publication years.

    Parameters
    ----------
    input_frame : pd.DataFrame
        The input DataFrame containing the time column.
    time_column : str, optional
        The name of the column containing publication dates, by default 'PublicationDate'.

    Returns
    -------
    pd.Series
        A series containing the extracted publication years.
    """
    assert time_column in list(input_frame.columns), f"Input frame must contain the specified time-column {time_column}"
    date_column = input_frame.loc[(input_frame.loc[:, time_column] != "Unknown")].loc[:, time_column]
    datetime_series = pd.to_datetime(date_column, format='ISO8601', errors='coerce')
    datetime_series.dropna(inplace=True)
    return datetime_series.dt.year.astype(int).rename('Publication Year')


def get_slice_publication_year(input_frame: pd.DataFrame, start_year: int = None, end_year: int = None, verbose=True,
                               time_column='PublicationDate') -> pd.DataFrame:
    """
    Filter a DataFrame to include articles published within a specified year range.

    This function slices the input DataFrame based on the publication year
    derived from a specified time column. It also provides an optional
    description of the filtered data.

    Parameters
    ----------
    input_frame : pd.DataFrame
        The input DataFrame containing the time column.
    start_year : int, optional
        The lower bound of the publication year range. If None, the minimum
        year in the data is used, by default None.
    end_year : int, optional
        The upper bound of the publication year range. If None, the maximum
        year in the data is used, by default None.
    verbose : bool, optional
        Whether to print a description of the filtered results, by default True.
    time_column : str, optional
        The name of the column containing publication dates, by default 'PublicationDate'.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only rows published within the specified year range.

    Raises
    ------
    AssertionError
        If the end year is less than or equal to the start year.
    """
    # derive publication year:
    temp_df = input_frame.copy()
    temp_df['Publication Year'] = get_publication_year(temp_df, time_column=time_column)
    # derive borders:
    lower_border = start_year if start_year is not None else temp_df['Publication Year'].min()
    upper_border = end_year if end_year is not None else temp_df['Publication Year'].max()
    assert upper_border > lower_border, f"end_year {upper_border} has to be higher than start_year {lower_border}!"
    # slice frame:
    time_slice = temp_df.loc[
        (temp_df['Publication Year'] <= upper_border) & (temp_df['Publication Year'] >= lower_border)]

    # return and eventually describe:
    if verbose:
        print(
            f"Selected {len(time_slice)} from {len(temp_df)} articles that were published{f' from {lower_border}' if start_year is not None else ''}{f' until {upper_border}' if end_year is not None else ''}!")
    return time_slice


######################## Training Evaluation Functions ########################
def get_pairwise_umass_coherence(model: LDAModel, word1_ind: int, word2_ind: int, verbose=False) -> float:
    """
    Calculates pairwise u-mass coherence, that describes how often words appear together compared to how often alone.
    Higher coherence scores are better, since these indicate inclusive topics.

    :param model: trained LDAModel to be analysed
    :param word1_ind: token for the first word
    :param word2_ind: token for the second word
    :param verbose: whether to print an explanatory statement

    :return: resulting u-mass coherence metric
    """
    # initialise counters:
    occurrence = co_occurrence = 0

    # iterate over documents and check for occurrences:
    for doc in model.docs:
        words = doc.words
        if word1_ind in words:
            occurrence += 1
            if word2_ind in words:
                co_occurrence += 1

    # explanatory statement:
    if verbose:
        print(f"{model.used_vocabs[word1_ind]} occurred {occurrence} times.")
        print(f"{model.used_vocabs[word1_ind]} and {model.used_vocabs[word2_ind]} co-occurred {co_occurrence} times.")

    # calculate metric:
    log_nominator = co_occurrence + 0.001  # increment to yield real numbers
    log_denominator = occurrence
    return math.log(log_nominator / log_denominator)


def get_topic_umass_coherence(model: LDAModel, topic_ind: int, words_to_consider: int = 3, verbose=False) -> float:
    """
    Calculated as the average pairwise coherence between all top-n words describing the topic.

    :param model: trained LDAModel to be analysed
    :param topic_ind: index of topic to be analysed
    :param words_to_consider: amount of words to be considered
    :return: resulting u-mass coherence metric
    """
    # retrieve top topic words:
    topic_words = model.get_topic_words(topic_ind, top_n=words_to_consider)
    topic_words = [word_prob[0] for word_prob in topic_words]  # neglect probability

    # explicatory statement:
    if verbose: print('Calculating u-mass coherence for words', topic_words)

    tokenization_dict = get_tokenization(model)  # to infer word indices

    # loop over all pairwise combinations and calculate pairwise coherence scores:
    coherence_list = []
    for word1, word2 in permutations(topic_words, 2):
        word1_ind, word2_ind = tokenization_dict[word1], tokenization_dict[word2]
        coherence_list.append(get_pairwise_umass_coherence(model, word1_ind, word2_ind))

    # return mean:
    return np.mean(coherence_list)


def get_model_umass_coherence(model: LDAModel, **umass_kwargs) -> float:
    """
    Return average topic-wise coherence for all topics of an LDAModel.

    :param model: model to be evaluated
    :param umass_kwargs: parameters passed to topic-wise coherence calculation

    :return: float with resulting u-mass coherence
    """
    umass_metrics = [get_topic_umass_coherence(model, topic_ind, **umass_kwargs) for topic_ind in range(model.k)]
    return np.mean(umass_metrics)


def plot_umass_scores(doc_list: [str], k_range: tuple = (10, 20, 2), iterations: int = 1000,
                      verbose=False, words_to_consider: int = 3, **lda_kwargs) -> pd.Series:
    """
    Evaluate an LDAModel's u-mass coherence score for different configurations (topic numbers) and plot the result.

    :param doc_list: list of strings (documents) to be included in model
    :param k_range: tuple with (start, stop, step) of topic numbers to consider
    :param iterations: amount of iterations to train each model before evaluating
    :param verbose: if True, shows an explicatory statement and progress bar during training
    :param words_to_consider: number of top words to consider for pair-wise coherence calculation per topics
    :param lda_kwargs: other (hyper-)parameters passed to LDAModel

    :return: pd.Series of resulting u-mass metrics per evaluated model
    """
    # iterate through configurations to be evaluated and save u-mass
    umass_metrics = []
    for k in tqdm(range(k_range[0], k_range[1], k_range[2])):
        model = LDAModel(k=k, **lda_kwargs)  # initialise model
        [model.add_doc(words=abstract.split()) for abstract in doc_list]  # add documents as single words
        if verbose: print(f"Training model with k={k}...")
        model.train(iterations, show_progress=verbose)  # train model
        umass_metrics.append(get_model_umass_coherence(model, words_to_consider=words_to_consider))

    # plot results:
    result = pd.Series(umass_metrics, index=range(k_range[0], k_range[1], k_range[2]), name="U-Mass Coherence")
    result.index.name = "Topic number"
    result.plot(title='Coherence Scores', ylabel=result.name, grid=True)
    return result


class TrainingLog:
    """
    This class is meant to be handed to LDAModel.train as callback in order to save the ll_per_word over training.
    """
    def __init__(self):
        self.iteration_list = list()
        self.ll_list = list()

    def __call__(self, model, current_iteration, total_iterations):
        self.iteration_list.append(current_iteration)
        self.ll_list.append(model.ll_per_word)

    @property
    def ll_over_iterations(self):
        return pd.Series(data=self.ll_list, index=self.iteration_list, name='LL-per-Word over Training')

    def plot(self):
        self.ll_over_iterations.plot(title='LDAModel Training Convergence', ylabel='Log Likelihood per Word',
                                     xlabel='Training Epoch')


######################## Modelling Workflow Functions ########################
def assign_topic_label_llm(word_list: list, generator: pipeline, long_prompt=True, verbose=True,
                           max_answer_tokens=5) -> str:
    """
    Leverage a pre-initialised LLM-pipeline from the hugging face library through prompt-engineering to label
    the topics found with LDA.

    :param word_list: list of word strings related to the topic
    :param generator: transformers.pipeline instance of type text generator
    :param long_prompt: if True, includes an example in the prompt
    :param verbose: if True, prints result
    :param max_answer_tokens: maximum number of tokens allowed in the generated answer

    :return: label assigned by the LLM
    """
    # initialise prompt parts:
    topic_str = "'" + "', '".join(word_list) + "'"
    pre_prompt = "You are a linguist. Please answer the following question with one or a composite word.\n"
    examples = """\nTwo examples: \nQuestion: To which topic do the words 'chair', 'table', 'couch', 'pillows' belong? Answer: furniture\nQuestion: To which topic do the words 'black', 'yellow', 'green', 'blue' belong? Answer: colors\n\nNow please answer the following question:"""
    question = f"To which topic do the words {topic_str} belong?"

    # construct prompt either with or without examples:
    prompt = pre_prompt + examples + question if long_prompt else pre_prompt + question

    # run and save answer
    answer = generator(prompt, max_new_tokens=max_answer_tokens)
    label = answer[0]['generated_text']

    # eventually print statement and return
    if verbose: print(f"Assigned {label} for {topic_str}!")
    return label


def label_topics_llm(model: LDAModel, generator: pipeline, no_of_words=8, long_prompt=True,
                     verbose=True) -> pd.DataFrame:
    """
    Method iterates over a word-per-topic distribution and labels each topic using an LLM.

    :param model: trained LDAModel with topic distribution to be labelled
    :param generator: transformers.pipeline instance of type text generator
    :param no_of_words: number of words per topic to be fed to the LLM
    :param long_prompt: if True, includes an example in the prompt
    :param verbose: if True, prints each label

    :return: pandas-dataframe containing the words and topic labels
    """
    # get word-topic-distribution:
    topic_word_dist = get_words_per_topic(model, no_of_words=no_of_words)

    # iterate over topics and save label:
    topic_labels = list()
    for topic_ind, words in tqdm(topic_word_dist.iterrows(), total=len(topic_word_dist)):
        topic_labels.append(assign_topic_label_llm(words[:-1], generator, verbose=verbose, long_prompt=long_prompt))

    # append to dataframe and return result:
    topic_word_dist['Topic'] = topic_labels
    topic_word_dist.set_index('Topic', inplace=True)
    return topic_word_dist


def label_validation_prompt(topic_frame: pd.DataFrame) -> str:
    """
    Generate a prompt for validating and optimizing topic labels.

    This function creates a formatted prompt string to assist in validating
    and improving topic labels. It includes examples and formatting rules,
    ensuring that each label matches the associated words and is unique.

    Parameters
    ----------
    topic_frame : pd.DataFrame
        A DataFrame where each row represents a topic. The first column
        contains the topic labels, and subsequent columns contain the words
        associated with each topic.

    Returns
    -------
    str
        A formatted prompt string to be used for manual label validation.
    """
    prompt = "Please see and optimise the following topic labels. The first word is the label that can be changed, the words behind it are input data belonging to that category. The first word has to match the rest. There may not be duplicates in the whole topic list.\nHere are two examples:\n'furniture': 'chair', 'table', 'couch', 'pillows'\n'colors': 'black', 'yellow', 'green', 'blue'\n\n Here comes the list:\n"

    for topic_label, words in topic_frame.iterrows():
        prompt += "'" + topic_label + "': '" + "', '".join(words[:-1]) + "'\n"

    prompt += "\nPlease only answer with the labels you want to correct and their position in the current list like '(2) cdk7 → cdk7 inhibitors' without any explanations!"
    return prompt


######################## Output Interpretation Functions ########################
def get_word_frequency(model: LDAModel) -> pd.DataFrame:
    """
    Retrieve word and document frequency data from an LDA model.

    Parameters
    ----------
    model : LDAModel
        The trained LDA model.

    Returns
    -------
    pd.DataFrame
        A DataFrame with word frequencies and document frequencies.
        Columns include:
        - 'Word Freq.': The total frequency of each word across topics.
        - 'Document Freq.': The number of documents in which each word appears.
    """
    return pd.DataFrame(index=list(model.used_vocabs),
                        data={'Word Freq.': model.used_vocab_freq, 'Document Freq.': model.used_vocab_df})


def get_words_per_topic(model: LDAModel, no_of_words=5) -> pd.DataFrame:
    """
    Retrieve the top words and their probabilities for each topic.

    Parameters
    ----------
    model : LDAModel
        The trained LDA model.
    no_of_words : int, optional
        The number of top words to retrieve for each topic, by default 5.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the top words for each topic and their probabilities.
        Columns include:
        - 'Top {n} Word': The nth most probable word for each topic.
        - 'Word Count': The number of documents assigned to each topic.
    """
    # initialise dictionaries with empty lists:
    top_word_dict = {index: list() for index in range(no_of_words)}
    probability_dict = {index: list() for index in range(no_of_words)}

    # fill lists:
    for k in range(model.k):
        for index, (word, probability) in enumerate(model.get_topic_words(k, top_n=no_of_words)):
            top_word_dict[index].append(word)
            probability_dict[index].append(probability)

    # change dict keys and add word count:
    top_word_dict = {f'Top {key + 1} Word': value for key, value in top_word_dict.items()}
    top_word_dict.update({"Word Count": model.get_count_by_topics()})

    return pd.DataFrame(data=top_word_dict)  # return as pd.DataFrame


def get_topic_assignments_for_corpus(model: LDAModel, input_strings: [str], topic_label_list: [str] = None):
    """
    Assign topics to a list of input strings based on an LDA model.

    Parameters
    ----------
    model : LDAModel
        The trained LDA model.
    input_strings : list of str
        A list of input text strings to assign topics.
    topic_label_list : list of str, optional
        Custom labels for topics. If None, numeric topic indices are used, by default None.

    Returns
    -------
    pd.Series or tuple
        If `topic_label_list` is None, returns a Series of topic numbers.
        Otherwise, returns a tuple containing two Series:
        - Topic numbers for each input string.
        - Topic labels for each input string.
    """
    # initialise documents:
    doc_subset = [model.make_doc(text.split()) for text in input_strings]

    # loop over docs and infer most probable topic:
    assigned_topic_numbers = [np.argsort(model.infer(doc)[0])[::-1][0] for doc in tqdm(
        doc_subset)]  # read first output (probabilities), sort in descending order and choose highest]
    assigned_topic_numbers_series = pd.Series(assigned_topic_numbers, index=range(len(input_strings)),
                                              name='Topic Number')

    # return with or without topic labels (depends on whether label list was provided):
    if topic_label_list is None:
        return assigned_topic_numbers_series
    else:
        assigned_topic_labels = [topic_label_list[i] for i in assigned_topic_numbers]
        assigned_topic_labels_series = pd.Series(assigned_topic_labels, index=range(len(input_strings)),
                                                 name='Topic Label')
        return assigned_topic_numbers_series, assigned_topic_labels_series


def get_topics_per_document(model: LDAModel, doc_index: int = None, doc: tomotopy.Document  = None,
                            topics_to_include: int = 5, verbose=False,
                            topic_name_list: [str] = None) -> pd.DataFrame:
    """
    Retrieve the top topics and their probabilities for a single document.

    Parameters
    ----------
    model : LDAModel
        The trained LDA model.
    doc_index : int, optional
        The index of the document in the model's corpus. Either `doc_index` or `doc` must be provided.
    doc : tomotopy.Document, optional
        A specific document. Either `doc_index` or `doc` must be provided.
    topics_to_include : int, optional
        The number of top topics to include, by default 5.
    verbose : bool, optional
        If True, prints the topic distribution of the document, by default False.
    topic_name_list : list of str, optional
        Custom labels for topics. If None, only numeric topic indices are used, by default None.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the top topics for the document and their probabilities.
    """
    # which document to use:
    assert doc_index is not None or doc is not None, "Either doc_index or doc (dominant) must be specified!"
    doc = model.docs[doc_index] if doc is None else doc

    # print doc eventually:
    if verbose:
        print(f"Topic distribution of document:")
        print(doc)

    # infer topic memberships:
    topic_dist = model.infer(doc)[0]  # topic distribution
    topic_memberships = np.argsort(topic_dist)[::-1][
                        :topics_to_include]  # topic number from highest to lowest membership prob

    # summarize as dataframe:
    if topic_name_list is None:
        temp_dict = {f'Top {ind + 1} topic': [topic_memberships[ind], topic_dist[topic_memberships[ind]]] for ind in
                     range(topics_to_include)}
        df = pd.DataFrame(data=temp_dict, index=['Topic Index', 'Probability'])
        # return formatted dataframe:
        return df.T.astype({'Topic Index': int, 'Probability': float})
    else:
        temp_dict = {f'Top {ind + 1} topic': [topic_memberships[ind], topic_name_list[topic_memberships[ind]],
                                              topic_dist[topic_memberships[ind]]] for ind in range(topics_to_include)}
        df = pd.DataFrame(data=temp_dict, index=['Topic Index', 'Topic Name', 'Probability'])
        # return formatted dataframe:
        return df.T.astype({'Topic Index': int, 'Topic Name': str, 'Probability': float})


def count_topic_occurrences(model: LDAModel, occurrence_threshold_topics=1, document_subset: [tomotopy.Document]=None,
                            topic_name_list: list = None):
    """
    Count the occurrences of each topic across a corpus or subset.

    Parameters
    ----------
    model : LDAModel
        The trained LDA model.
    occurrence_threshold_topics : int, optional
        Minimum number of top topics to consider per document, by default 1.
    document_subset : list of tomotopy.Document, optional
        A subset of documents to evaluate. If None, the entire corpus is used, by default None.
    topic_name_list : list of str, optional
        Custom labels for topics. If None, only numeric topic indices are used, by default None.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing topic occurrence counts.
    """
    if topic_name_list is not None:
        occurrence_frame = pd.DataFrame(index=pd.Series(range(model.k), name='Topic Number'),
                                        data={"Occurrences": [0] * model.k, "Label": topic_name_list})
    else:
        occurrence_frame = pd.DataFrame(index=pd.Series(range(model.k), name='Topic Number'),
                                        data={"Occurrences": [0] * model.k})

    if document_subset is None:  # either iterate over all documents present in the model:
        for doc_ind in range(len(model.docs)):
            predicted_topics = get_topics_per_document(model, doc_index=doc_ind, topics_to_include=occurrence_threshold_topics)[
                'Topic Index']
            for predicted_topic in predicted_topics:
                occurrence_frame.loc[predicted_topic, 'Occurrences'] += 1
    else:  # or iterate over a provided document subset:
        for doc in document_subset:
            predicted_topics = get_topics_per_document(model, doc=doc, topics_to_include=occurrence_threshold_topics)[
                'Topic Index']
            for predicted_topic in predicted_topics:
                occurrence_frame.loc[predicted_topic, 'Occurrences'] += 1

    return occurrence_frame


def count_topic_co_occurrences(model: LDAModel, co_occurrence_threshold_topics=3,
                               document_subset: [tomotopy.Document]=None) -> pd.DataFrame:
    """
    Count co-occurrences of topics and return a co-occurrence matrix.

    Parameters
    ----------
    model : LDAModel
        The trained LDA model.
    co_occurrence_threshold_topics : int, optional
        The number of top topics to consider per document for co-occurrence, by default 3.
    document_subset : list of tomotopy.Document, optional
        A subset of documents to evaluate. If None, the entire corpus is used, by default None.

    Returns
    -------
    pd.DataFrame
        A DataFrame representing the co-occurrence matrix. Only the upper triangle is filled.
    """
    # initialise matrix with topic indices as rows and columns and zeros in the upper right tri-angle (elsewhere NaN):
    co_occurrence_frame = pd.DataFrame(index=pd.Series(range(model.k), name='Topic Number'),
                                       data={ind: [0] * ind + [np.nan] * (model.k - ind) for ind in range(model.k)})

    if document_subset is None:  # either iterate over all documents present in the model:
        for doc_ind in range(len(model.docs)):
            # we predict the top topics per document:
            predicted_topics = get_topics_per_document(model, doc_index=doc_ind, topics_to_include=co_occurrence_threshold_topics)[
                'Topic Index']

            # then iterate over all combinations (not permutations, to ignore reversals and thereby prevent errors from double considerations)
            # and increment the respective cell to indicate a co-occurrence
            for topic_index1, topic_index2 in combinations(predicted_topics, 2):
                # we only want to fill the upper right tri-angle of the matrix to avoid redundancy, this is accomplished by
                # sorting the topic_indices, using the higher as column index and the lower as row index
                row_index = min(topic_index1, topic_index2)
                column_index = max(topic_index1, topic_index2)
                co_occurrence_frame.iloc[row_index, column_index] += 1

    else:  # or iterate over a provided document subset:
        for doc in document_subset:  # procedure similar as above
            predicted_topics = get_topics_per_document(model, doc=doc, topics_to_include=co_occurrence_threshold_topics)['Topic Index']
            for topic_index1, topic_index2 in combinations(predicted_topics, 2):
                row_index = min(topic_index1, topic_index2)
                column_index = max(topic_index1, topic_index2)
                co_occurrence_frame.iloc[row_index, column_index] += 1

    return co_occurrence_frame


def get_communities(G: nx.Graph, algorithm: Literal["greedy", "leiden", "multilevel"] = "greedy", verbose=True,
                    **algorithm_kwargs) -> dict:
    """
    Detect communities in a network graph and return a corresponding membership dict.
    See https://igraph.org/c/html/0.9.4/igraph-Community.html#igraph_community_leiden.

    :param G: networkx graph object
    :param algorithm: algorithm to use, can be "greedy", "leiden" or "multilevel"
    :param verbose: if True, prints result overview upon execution
    :param **algorithm_kwargs: kwargs passed to the igraph algorithm implementations

    :return: dict with structure {community_index: [node1, node2, ...], ...}
    """
    # parse networkx graph to igraph:
    edges = list(G.edges())
    edge_weights = list(nx.get_edge_attributes(G, 'weight').values())
    node_weights = list(nx.get_node_attributes(G, 'weight').values())
    G_ig = ig.Graph(edges=edges, directed=G.is_directed(),
                    edge_attrs={'weight': edge_weights},
                    vertex_attrs={'weight': node_weights})

    # run algorithm:
    if algorithm == "greedy":  # fast greedy communities, see Clauset et Al. "Finding community structures in very large networks" (2004)
        clusters = G_ig.community_fastgreedy(weights=edge_weights, **algorithm_kwargs).as_clustering()  # run algo
        membership_dict = {i: cluster for i, cluster in enumerate(clusters)}  # construct dict

    elif algorithm == 'leiden':  # leiden community detection algorithm
        memberships = G_ig.community_leiden(weights=edge_weights, **algorithm_kwargs).membership  # run algo
        membership_dict = dict()  # construct dict
        for node, community in enumerate(memberships):
            if community not in membership_dict.keys():
                membership_dict[community] = [node]
            else:
                membership_dict[community].append(node)

    elif algorithm == 'multilevel':  # optimize modularity in multiple levels
        memberships = G_ig.community_multilevel(weights=edge_weights, **algorithm_kwargs).membership  # run algo
        membership_dict = dict()  # construct dict
        for node, community in enumerate(memberships):
            if community not in membership_dict.keys():
                membership_dict[community] = [node]
            else:
                membership_dict[community].append(node)

    else:
        raise ValueError(f"Algorithm '{algorithm}' not recognized")

    # return and eventually print results
    if verbose:
        for community, members in membership_dict.items():
            print(f"Community {community}: {members}")
    return membership_dict


######################## Output Plotting Functions ########################
def plot_word_clouds(model: LDAModel, cloud_word_count=50, topics_to_evaluate: list = None,
                     topic_labels : [str] = None) -> None:
    """
    Plot word clouds for selected topics from an LDA model.

    This function visualizes the most frequent words in selected topics
    using word clouds. Each topic's words are scaled based on their
    probabilities.

    Parameters
    ----------
    model : LDAModel
        The trained LDA model containing the topic-word distributions.
    cloud_word_count : int, optional
        The maximum number of words to include in each word cloud, by default 50.
    topics_to_evaluate : list, optional
        A list of topic indices to visualize. If None, all topics are visualized, by default None.
    topic_labels : list of str, optional
        Custom labels for topics. Must match the length of `topics_to_evaluate`
        if provided. By default, topics are labeled as "Topic {index}".

    Raises
    ------
    ValueError
        If custom topic labels are provided and their count does not match
        the length of `topics_to_evaluate`.
    """
    # plot word_clouds:
    if topics_to_evaluate is None:
        topics_to_evaluate = list(range(model.k))
    # prevent errors:
    if topic_labels is not None:
        if len(topic_labels) < len(topics_to_evaluate):
            raise ValueError("If providing custom topic labels, these need to fit the length of topics_to_evaluate!")
    else:
        topic_labels = [f"Topic {topic_ind}" for topic_ind in range(len(topics_to_evaluate))]

    # initialise plots:
    fig, axes = plt.subplots(1, len(topics_to_evaluate), figsize=(2.5 * len(topics_to_evaluate), 20))
    axes = axes.flatten()

    # iterate over topics:
    for ax_ind, topic_ind in enumerate(topics_to_evaluate):
        # define frequency as probability:
        words_and_probs = model.get_topic_words(topic_ind, top_n=cloud_word_count)
        word_freq = {word: prob for word, prob in words_and_probs}

        # generate wordcloud from frequency:
        wordcloud = WordCloud(width=400, height=400, background_color="white").generate_from_frequencies(word_freq)

        # plotting and formatting:
        axes[ax_ind].imshow(wordcloud, interpolation="bilinear")
        axes[ax_ind].axis("off")
        axes[ax_ind].set_title(topic_labels[ax_ind])

    # format and display:
    plt.tight_layout()
    plt.show()


def plot_publications_per_year(article_frame: pd.DataFrame, time_column: str = 'PublicationDate',
                               starting_year: int = None, ending_year: int = None, years_to_summarize: int = 1) -> None:
    """
    Plot a histogram of publications over time.

    This function visualizes the distribution of publication counts
    over the years, optionally grouping years into bins.

    Parameters
    ----------
    article_frame : pd.DataFrame
        The input DataFrame containing publication data.
    time_column : str, optional
        The name of the column containing publication dates, by default 'PublicationDate'.
    starting_year : int, optional
        The starting year for the plot. If None, the minimum year in the data is used, by default None.
    ending_year : int, optional
        The ending year for the plot. If None, the maximum year in the data is used, by default None.
    years_to_summarize : int, optional
        The number of years to group together in each bin, by default 1.
    """
    # infer publication years:
    publication_year = get_publication_year(article_frame, time_column)

    # define year bins for histogram:
    start_x = publication_year.min() if starting_year is None else starting_year
    end_x = publication_year.max() if ending_year is None else ending_year
    binsequence = list(range(start_x, end_x, years_to_summarize)) + [end_x]

    # plot histogram:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(x=publication_year, bins=binsequence, color='blue', edgecolor='black', alpha=0.7)
    ax.set_title(f"Publications per {str(years_to_summarize) + '-' if years_to_summarize > 1 else ''}year{'s' if years_to_summarize > 1 else ''}")
    ax.set_ylabel("Number of Publications")
    ax.set_xlabel("Year")
    plt.show()


def initialise_topic_graph(model: LDAModel, occurrence_threshold_topics=1, co_occurrence_threshold_topics=2,
                           article_subset: pd.DataFrame = None, topic_labels: list = None,
                           mpl_colorpalette="tab20c", edge_color="grey",
                           node_scale_factor=30, edge_scale_factor=None, edge_power=1.4,
                           optimal_node_distance=2.1, title:str=None, save_title=None, hidden=False):
    """
    Initialize a topic co-occurrence graph from an LDA model.

    This function creates a network graph where nodes represent topics,
    and edges represent co-occurrence relationships. The graph can be customized
    based on topic occurrence thresholds, co-occurrence thresholds, and visual
    parameters.

    Parameters
    ----------
    model : LDAModel
        The trained LDA model containing topic data.
    occurrence_threshold_topics : int, optional
        Minimum number of times a topic must appear to be included in the graph, by default 1.
    co_occurrence_threshold_topics : int, optional
        Minimum number of co-occurrences between topics to include an edge, by default 2.
    article_subset : pd.DataFrame, optional
        A subset of articles to evaluate. If provided, only this subset is used to calculate
        occurrences and co-occurrences, by default None.
    topic_labels : list, optional
        Custom labels for topics. If None, topics are labeled with their indices, by default None.
    mpl_colorpalette : str, optional
        Name of a Matplotlib colormap to color nodes, by default "tab20c".
    edge_color : str, optional
        Color of the edges in the graph, by default "grey".
    node_scale_factor : int, optional
        Scaling factor for node sizes, by default 30.
    edge_scale_factor : float, optional
        Scaling factor for edge widths. If None, defaults are set based on the co-occurrence threshold, by default None.
    edge_power : float, optional
        Exponent to adjust edge weights, by default 1.4.
    optimal_node_distance : float, optional
        Optimal distance between nodes in the graph layout, by default 2.1.
    title : str, optional
        Title of the graph plot, by default None.
    save_title : str, optional
        Path to save the graph plot. If None, the plot is not saved, by default None.
    hidden : bool, optional
        Whether to suppress graph plotting and preview, by default False.

    Returns
    -------
    tuple
        A tuple containing the graph object (networkx.Graph) and the node position dictionary.

    Examples
    --------
    >>> G, pos = initialise_topic_graph(lda_model, occurrence_threshold_topics=2, co_occurrence_threshold_topics=3)
    >>> plot_topic_graph(G, pos)
    """
    # eventually create tomotopy.Document list from article_subset
    if article_subset is not None:
        print(f'Article subset with {len(article_subset)} articles provided. Parsing such to model...')
    doc_subset = [model.make_doc(article.split()) for article in
                  article_subset.Abstract] if article_subset is not None else None

    # calculate occurrences and co-occurrences:
    print('Calculating (co)-occurrences...')
    occurrence_frame = count_topic_occurrences(model, occurrence_threshold_topics, document_subset=doc_subset)
    co_occurrence_frame = count_topic_co_occurrences(model, co_occurrence_threshold_topics, document_subset=doc_subset)

    # initialise graph:
    G = nx.Graph()

    # initialise nodes:
    amount_of_nodes = model.k
    colormap = plt.get_cmap(mpl_colorpalette, amount_of_nodes)
    color_list = [colormap(i) for i in range(amount_of_nodes)]
    size_list = list(occurrence_frame['Occurrences'])
    # iterate over sizes and colors and add to graph:
    for index, (color, weight) in enumerate(zip(color_list, size_list)):
        G.add_node(index, color=color, weight=weight * node_scale_factor)

    # initialise edges:
    for topic1 in co_occurrence_frame.columns:
        # iterate over top-right of matrix:
        for topic2 in range(topic1):
            co_occurrence = co_occurrence_frame.iloc[topic2, topic1]  # the order of iloc is important!
            if co_occurrence == 0: continue  # skip non-co-occurring topics
            # scale edge weights depending on no of topics to consider for co-occurrence (because this is ~to number of edges)
            edge_scale_factor = (1/200 if co_occurrence_threshold_topics > 2 else 1/50) if edge_scale_factor is None else edge_scale_factor

            # add weighted edges for co-occurring topics:
            G.add_weighted_edges_from(
                ebunch_to_add=[(topic1, topic2, (co_occurrence ** edge_power) * edge_scale_factor)], color=edge_color)

    # initialise positions:
    # spring layout creates a force-based representation treating nodes as anti-gravitational objects and edges as springs
    # k describes the optimal distance between nodes
    pos = nx.spring_layout(G, k=optimal_node_distance,
                           seed=42)  # options are spring_layout, circular_layout, shell_layout, spectral_layout

    # eventually plot graph and print explanatory statements:
    if not hidden:
        print("Topic graph initialised! Printing preview...")
        plot_topic_graph(G, pos, topic_labels=topic_labels, title=title, save_title=save_title)
    print("This function returns a tuple with G (graph object) and pos (node position-dict).")
    print("It is recommended, to manually adjust the positions from the position-dict to improve the look of the plot.")
    print("Afterwards, the plot can be recreated with plot_topic_graph(G, pos)")

    # return objects
    return G, pos


def plot_topic_graph(G, pos, topic_labels: list = None,
                     fig_size=(16, 16), legend_pos=(0.5, -0.4),
                     title: str = None, save_title=None, legend_cols: int = None,
                     node_colors: Union[list, dict] = None, edge_colors: Union[list, dict] = None,
                     community_memberships: dict = None, include_community_labels=True) -> None:
    """
    Plot a topic graph using NetworkX and matplotlib.

    :param G: networkx graph object
    :param pos: networkx positional encoding of nodes
    :param topic_labels: list of topic labels, has to match the order of nodes
    :param fig_size: plot size
    :param legend_pos: position of legend in relative coordinates (x, y)
    :param title: if provided, is displayed at top of figures, otherwise default title is used
    :param save_title: if provided, save fig to such
    :param node_colors: dict {node: color, ...} or list with node colors to be used
    :param edge_colors: dict {(node1, node2): color, ...} or list with edge colors to be used
    :param community_memberships: dict with structure {community_name: [node1, node2, ...], ...} used to order legend
    :param include_community_labels: if True, label communities in legend
    """
    # infer properties to create graph:
    node_weights = list(nx.get_node_attributes(G, 'weight').values())
    edge_weights = list(nx.get_edge_attributes(G, 'weight').values())
    # infer colors from graph or argument. in latter case, dicts and lists are allowed as input:
    node_colors = list(nx.get_node_attributes(G, 'color').values()) if node_colors is None else (
        [node_colors[node] for node in G.nodes] if isinstance(node_colors, dict) else node_colors)
    edge_colors = list(nx.get_edge_attributes(G, 'color').values()) if edge_colors is None else (
        [edge_colors[edges] for edges in G.edges] if isinstance(edge_colors, dict) else edge_colors)
    # prevent errors:
    assert len(G.nodes) == len(node_weights) == len(
        node_colors), f"Node attribute list flawed! {len(G.nodes)} != {len(node_weights)} != {len(node_colors)}"
    assert len(G.edges) == len(edge_weights) == len(
        edge_colors), f"Edge attribute list flawed! {len(G.edges)} != {len(edge_weights)} != {len(edge_colors)}"

    # plotting:
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=node_weights, node_color=node_colors, edge_color=edge_colors,
            width=edge_weights, font_size=15)
    # distinct title considering title and community_memberships:
    ax.set_title(("Topic Network Graph" if community_memberships is None else "Topic Network Graph with Communities") if title is None else title,
                 fontsize=30)

    # legend:
    if topic_labels is not None:
        if community_memberships is None:  # standard legend with community labels
            label_list = [f"({ind}) {topic_labels[ind]}" for ind in range(len(topic_labels))]
            legend_elements = [
                Line2D([], [], color="white", marker='o', label=label_list[node_ind],
                       markerfacecolor=node_colors[node_ind], markersize=15) for node_ind in range(len(node_colors))
            ]

        else:  # amend legend to convey community memberships more clearly:
            legend_elements = []  # initialise empty list
            for ind, (community, members) in enumerate(community_memberships.items()):
                # add space between communities' entries:
                if ind != 0: legend_elements.append(
                    Line2D([], [], color="white", label=""))
                # add community label before communities' entries:
                if include_community_labels: legend_elements.append(
                    Line2D([], [], color="white", label=f"--- Community {community} ---"))
                # jointly add community members to legend:
                for node in members:
                    legend_elements.append(
                        Line2D([], [], color="white", marker='o', label=f"({node}) {topic_labels[node]}",
                               markerfacecolor=node_colors[node], markersize=15))

        # display legend:
        plt.legend(handles=legend_elements, loc='lower center',
                   ncols=(len(node_colors) // 15 if legend_cols is None else legend_cols),
                   bbox_to_anchor=legend_pos, fontsize=15)

    # saving:
    if save_title is not None:
        plt.savefig(save_title, bbox_inches='tight')


def get_community_edge_node_colors(G, community_memberships: dict, mpl_colormap: Literal["jet","rainbow","gist_rainbow","turbo"] = "jet") -> (dict, dict):
    """
    Assign colors to edges and nodes from a NetworkX graph based on communities.

    :param G: networkx graph object
    :param community_memberships: dict with structure {community_name: [node1, node2, ...], ...}
    :param mpl_colormap: matplotlib colormap to be used

    :return: tuple with two dicts, first {edge: color}, second {node: color}
    """
    # assign colors to communities:
    colormap = plt.get_cmap(mpl_colormap, len(community_memberships.keys()))  # rainbow, gist_rainbow, turbo
    community_color_dict = {i: colormap(i) for i in range(len(community_memberships.keys()))}

    # edges within one community get the community color:
    edge_color_dict = nx.get_edge_attributes(G, 'color')
    for node1, node2 in G.edges:
        for community, members in community_memberships.items():
            if node1 in members:
                if node2 in members:
                    # if both nodes in same community:
                    edge_color_dict[(node1, node2)] = community_color_dict[community]
                else:
                    # if second node not in same we can abort the inner loop, because the first node won't occur in another one
                    break

    # nodes receive their community color:
    node_color_dict = nx.get_node_attributes(G, 'color')
    for node in G.nodes:
        for community, members in community_memberships.items():
            if node in members:
                node_color_dict[node] = community_color_dict[community]

    # return dicts:
    return edge_color_dict, node_color_dict