import pandas as pd
import os
from tqdm import tqdm
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from collections import Counter


class TextCorpusPreprocessor:
    """
    A class to import and preprocess a text corpus from csv- or json-files.

    The class provides methods to clean, transform, and preprocess a text corpus by performing tasks such as
    removing punctuation, stop words, and infrequent words, converting text to lowercase, lemmatizing words,
    replacing abbreviations, marking frequent n-grams, and more.

    Attributes
    ----------
    text_frame : pd.DataFrame
        DataFrame containing the imported and cleaned text articles.
    text_column : str
        Column name in `text_frame` that holds the text data.
    raw_text_frame : pd.DataFrame
        A copy of the original `text_frame` before any preprocessing.
    """
    def __init__(self,
                 input_directory: str,
                 exclusion_file:str=None,
                 id_column='PMID',
                 text_column='Abstract'
                 ):
        """
        Initialize a `TextCorpusPreprocessor` instance by importing articles and cleaning missing values.

        This method loads the corpus from the specified directory, imports articles into a DataFrame, and handles
        missing or empty data.

        Parameters
        ----------
        input_directory : str
            Directory containing the corpus files.
        exclusion_file : str, optional
            Path to a file with articles to exclude, by default None.
        id_column : str, optional
            The name of the column containing article IDs, by default 'PMID'.
        text_column : str, optional
            The name of the column containing the text data, by default 'Abstract'.
        """
        self.text_frame = self.import_articles(input_directory, exclusion_file, verbose=False, id_column=id_column)
        self.text_column = text_column
        self.drop_missing_data()  # drop and fill na
        self.raw_text_frame = self.text_frame.copy()

    def __repr__(self):
        """ String representation """
        return f"TextCorpusPreprocessor instance with {len(self.text_frame)} articles."

    def __str__(self):
        """ Print operator """
        return self.__repr__()

    @property
    def text_series(self) -> pd.Series:
        """
        Get the text entries from the `text_frame` DataFrame.

        Returns
        -------
        pd.Series
            A series containing the text data from the specified column.
        """
        return self.text_frame.loc[:, self.text_column]

    @text_series.setter
    def text_series(self, value):
        """
        Set the text entries in the `text_frame` DataFrame.

        Parameters
        ----------
        value : pd.Series
            The new text data to set in the `text_column` of `text_frame`.
        """
        self.text_frame.loc[:, self.text_column] = value

    def drop_missing_data(self):
        """
        Drop missing texts and fill NaN values with empty strings in the text column.

        This method removes rows where the text is missing or contains only spaces.
        """
        self.text_frame = self.text_frame.loc[(self.text_series != "") &
                                                    (self.text_series != " ")]
        self.text_frame.dropna(inplace=True, subset=self.text_column)
        self.text_frame.fillna(inplace=True, value="")

    def convert_to_lower_case(self):
        """ Convert all texts into lower case. """
        self.text_series = self.text_series.apply(lambda x: x.lower())

    def remove_punctuation(self):
        """
        Remove punctuation characters from the text entries.

        This method handles punctuation in two categories:
        - Removing certain punctuation characters anywhere in the text.
        - Removing punctuation only when it occurs at the end of words.
        """
        # some punctuation characters do not necessarily occur at the end of words, these get removed independent of where they are:
        punctuation_chars_anywhere = ["[", "]", "(", ")", "\"", "'", "+", "<", ">", "%", "•", "º", "±", "~", "$", "…", "=", "*"]
        self.text_series = self.text_series.apply(self.replace_letters, candidate_list=punctuation_chars_anywhere)

        # others should only be removed, if at the end of a word:
        punctuation_chars_suffix = [",", ".", "!", "?", ";", ":", "_"]
        # list here  is without dash (-), because we re-use this list later, replacing such chars mid-word assuming they
        # indicate a missing white space. such assumption doesn't seem valid for dashs, because these mostly indicate a compound word
        self.text_series = self.text_series.apply(self.replace_suffix, candidate_list=punctuation_chars_suffix + ["-"], verbose=False)

        # there remain some issues with either links, missing spaces or decimal numbers:
        self.text_series = self.text_series.apply(self.remove_numbers)  # remove numbers
        # condition to check for links:
        def is_link(input_string: str) -> bool:
            # character-sequences indicating a link:
            link_indicators = ["https://", "http://", "www."]
            for indicator in link_indicators:
                if indicator in input_string:
                    return True
            # if none of the indicators was found, return False:
            return False
        # replace the remaining punctuation of words that are no links with whitespaces:
        self.text_series = self.text_series.apply(self.replace_letters, candidate_list=punctuation_chars_suffix,
                                                  replace_with=" ", omit_condition=is_link)

    def remove_stop_words(self, custom_stop_word_list: [str] = None):
        """
        Remove common stop words from the text entries.

        Parameters
        ----------
        custom_stop_word_list : list of str, optional
            A custom list of stop words to remove, by default None (uses predefined stop words).
        """
        stop_word_list = "a, about, above, after, again, against, ago, ah, all, almost, along, already, also, although, always, am, an, and, another, any, anybody, anyhow, anyone, anything, anywhere, are, aren't, around, as, aside, at, away, back, be, became, because, become, becomes, been, before, beforehand, being, below, beside, besides, between, beyond, both, but, by, came, can, can't, cannot, come, comes, couldn't, could, did, didn't, do, does, doesn't, doing, don't, done, down, during, each, either, else, enough, even, ever, every, everybody, everyone, everything, everywhere, except, few, following, for, former, formerly, forth, found, from, further, get, gets, getting, go, goes, going, gone, got, had, hadn't, has, hasn't, have, haven't, having, he, he'd, he'll, he's, hence, her, here, here's, hers, herself, him, himself, his, how, how's, however, i, i'd, i'll, i'm, i've, if, in, indeed, into, is, isn't, it, it's, its, itself, just, keep, kept, know, known, knows, last, later, least, let, let's, like, likely, likewise, little, look, looking, looks, lot, made, make, makes, many, may, maybe, me, might, mine, more, moreover, most, mostly, much, must, mustn't, my, myself, near, nearly, need, needed, needing, needs, neither, never, nevertheless, next, no, nobody, none, nor, not, nothing, now, nowhere, of, off, often, on, once, one, only, onto, or, other, others, otherwise, ought, our, ours, ourselves, out, over, own, part, past, perhaps, placed, please, quite, rather,really, regarding, respectively, right, said, same, say, says, see, seen, seem, seemed, seeming, seems, several, shall, shan't, she, she'd, she'll, she's, should, shouldn't, since, so, some, somebody, somehow, someone, something, sometime, sometimes, somewhere, still, such, sure, take, taken, taking, than, that, that's, the, their, theirs, them, themselves, then, there, there's, therefore, these, they, they'd, they'll, they're, they've, thing, things, think, this, those, though, through, throughout, thus, to, too, took, toward, towards, under, until, up, upon, us, use, used, uses, using, usually, vs, very, was, wasn't, way, we, we'd, we'll, we're, we've, well, went, were, weren't, what, what's, whatever, when, when's, whenever, where, where's, wherever, whether, which, while, who, who's, whom, whose, why, why's, will, with, within, without, won't, would, wouldn't, yes, yet, you, you'd, you'll, you're, you've, your, yours, yourself, yourselves, -, b, c, d, e, f, g, h, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, ii, iii, iv".split(
            ", ") if custom_stop_word_list is None else custom_stop_word_list
        # we need to construct this list without punctuation, because apostrophes are removed during punctuation removal:
        stop_word_list_without_apostrophes = [self.replace_letters(word, ["'"]) for word in stop_word_list]
        # now remove such words:
        self.text_series = self.text_series.apply(self.replace_words, candidate_list=stop_word_list_without_apostrophes)

    def mark_n_grams(self, **n_gram_kwargs):
        """
        Mark frequent n-grams with underscores.

        This method detects frequent n-grams and replaces them in the text with underscores to indicate compound words.

        Parameters
        ----------
        n_gram_kwargs : dict
            Additional arguments for the n-gram detection function (e.g., n-gram size, occurrence threshold).
        """
        n_gram_dict = self.detect_n_grams(self.text_series, **n_gram_kwargs)  # detect frequent n-grams
        self.text_series = self.text_series.apply(self.connect_compound_words,
                                                  candidate_list=n_gram_dict.keys())  # mark such with underscores

    def replace_abbreviations(self, abbreviation_txt_file: str):
        """
        Replace abbreviations in the text with their full meanings.

        This method uses a provided text file that maps abbreviations to their meanings and performs the replacement.

        Parameters
        ----------
        abbreviation_txt_file : str
            Path to a text file containing abbreviation-meaning pairs.
        """
        abbreviation_mapping = self.construct_abbreviation_dict(abbreviation_txt_file)  # construct mapping
        self.text_series = self.text_series.apply(self.replace_words, candidate_mapping=abbreviation_mapping)  # replace abbreviations
        # mark meanings as compound words through underscores:
        self.text_series = self.text_series.apply(self.connect_compound_words,
                                                  candidate_list=abbreviation_mapping.values())

    def lemmatize_texts(self, custom_typical_endings: [str] = None, occurrence_threshold=1,
                        inplace=True, additional_infeasible_candidates: [str] = None):
        """
        Lemmatize the texts by detecting word stems and replacing word endings.

        Parameters
        ----------
        custom_typical_endings : list of str, optional
            Custom list of word endings to treat as typical endings for lemmatization, by default None.
        occurrence_threshold : int, optional
            Minimum occurrence threshold for lemmatization to be considered valid, by default 1.
        inplace : bool, optional
            Whether to apply the changes directly to the text corpus, by default True.
        additional_infeasible_candidates : list of str, optional
            Additional words to exclude from lemmatization, by default None.
        """
        # define endings to be replaced with *:
        typical_endings = ['ed', 'ing', 'ments', 'er'] if custom_typical_endings is None else custom_typical_endings
        # define words to skip because they have no / are the word stem:
        infeasible_candidates = ['blessed', 'beloved', 'building', 'painting', 'setting', 'whether', 'upper', 'lower',
                                 'hundred', 'answer', 'center', 'after', 'younger', 'however', 'september', 'december',
                                 'ai-based', 'younger', 'paillier', 'based', 'selected', 'outdated', 'suffer',
                                 'elsevier', 'parameter', 'server', 'scaffolding', 'paper', 'render', 'liver', 'layer',
                                 'wording', 'counter'] + ([] if additional_infeasible_candidates is None else additional_infeasible_candidates)
        print(f"Removing {typical_endings} to detect word stems.\nOmitting {infeasible_candidates}.")

        # calculate provisional lemmatization:
        stem_origin_dict = dict()  # will be filled with amended words and origins during runtime:
        # output can be omitted, we only require the stem_origin_dict:
        _ = self.text_series.apply(self.replace_suffix, candidate_list=typical_endings,
                                   replace_with="*", verbose=False, log_dict=stem_origin_dict)

        # we exclude redundant stems that originate only from only one (or an arbitrary threshold amount of) word(s):
        redundant_candidates = list()  # the origin list is necessary to omit them in the next lemmatization step
        redundant_stems = list()  # the stem list is to ignore such during further scrutiny (see next step)
        # iterate and find redundant stems:
        for stem, origin_list in stem_origin_dict.items():
            # we use a set to prevent duplicates and test whether its length is only one, i.e. only one origin is present:
            if len(set(origin_list)) == occurrence_threshold:
                # add to lists
                redundant_candidates += [origin_list[0]]
                redundant_stems += [stem]

        # check function serving as omit-condition:
        def infeasible_lemmatization(input_string: str) -> bool:
            infeasible_set = set(infeasible_candidates + redundant_candidates)
            return input_string in infeasible_set

        # now final lemmatization omitting redundant and infeasible ones:
        stem_origin_dict = dict()
        result = self.text_series.apply(self.replace_suffix, candidate_list=typical_endings, replace_with="*",
                                        omit_condition=infeasible_lemmatization, verbose=False, log_dict=stem_origin_dict)

        # count occurrences :
        stem_count_dict = {stem: len(stem_origin_dict[stem]) for stem in set(stem_origin_dict.keys())}

        # print output for eventual scrutiny:
        for stem, count in stem_count_dict.items():
            print(f"{stem} occurring {count} times. Derived from:")
            print(stem_origin_dict[stem], "\n")

        # if inplace, save result:
        if inplace:
            self.text_series = result
        else:
            print("Result was not saved but returned. Set inplace=True to save.")
            return result

    def remove_infrequent_words(self, tolerance=0.01, inplace=True):
        """
        Remove infrequent words based on a given tolerance.

        Parameters
        ----------
        tolerance : float, optional
            The minimum frequency ratio for a word to be kept, by default 0.01 (1%).
        inplace : bool, optional
            Whether to apply the changes directly to the text corpus, by default True.
        """
        result = self.text_series.apply(self.remove_words_below_tolerance_from_string, tolerance=tolerance)
        if inplace:
            self.text_series = result
        else:
            print("Result was not saved but returned. Set inplace=True to save.")
            return result

    def full_preprocessing_workflow(self, abbreviation_txt_file: str = None):
        """
        Run the entire preprocessing workflow on the text corpus.

        This includes steps such as converting to lowercase, removing punctuation,
        stop words, and infrequent words, marking n-grams, and lemmatization.

        Parameters
        ----------
        abbreviation_txt_file : str, optional
            Path to a text file containing abbreviation-meaning pairs to replace abbreviations, by default None.
        """
        self.convert_to_lower_case()
        self.remove_punctuation()
        self.remove_stop_words()
        self.mark_n_grams()
        if abbreviation_txt_file is not None: self.replace_abbreviations(abbreviation_txt_file)
        self.lemmatize_texts()
        self.remove_infrequent_words()

    def export_json(self, output_path: str):
        """
        Export the preprocessed text corpus to a JSON file.

        Parameters
        ----------
        output_path : str
            Path where the preprocessed data will be saved as a JSON file.
        """
        self.text_frame.to_json(output_path)

    ########## static auxiliary methods: ##########
    @staticmethod
    def import_articles(article_dir, exclusion_file=None, verbose=True, id_column:str = "PMID"):
        """
        Import articles while omitting such defined on an exclusion txt-file. Can handle csv and json input.

        :param article_dir: folder with json or csv article frames
        :param exclusion_file: exclusion file with lines structured as [id to exclude] -- [reason]
        :param verbose: whether to print excluded articles + reason
        :param id_column: name of identifier column
        :return: dataframe with imported articles
        """
        # read exclusion list:
        id_blacklist_dict = dict()
        if exclusion_file is not None:
            with open(exclusion_file) as f:
                for line in f.readlines():
                    if line[0] == "#": continue  # skip comments
                    if line.strip() == "": continue  # skip empty lines
                    # read ids to be excluded:
                    try:
                        id, reason = line.split("--")
                    except ValueError:
                        id = line
                        reason = "no reason provided"

                    # remove spaces and add to blacklist:
                    id = int(id.strip())
                    reason = reason.strip()
                    id_blacklist_dict.update({id: reason})

                    # explanatory statement:
                    if verbose: print(f"Excluding {id} for reason: {reason}.")

        # import articles from article directory:
        article_frame = pd.DataFrame()
        for file_name in tqdm(os.listdir(article_dir)):
            if file_name.endswith(".json"):
                temp_article_frame = pd.read_json(article_dir / file_name)  # read JSON
            elif file_name.endswith(".csv"):
                temp_article_frame = pd.read_csv(article_dir / file_name)  # read CSV
            else:
                continue  # skip non-json/csv files
            temp_article_frame = temp_article_frame.loc[
                ~(temp_article_frame.loc[:, id_column].isin(id_blacklist_dict.keys()))]  # consider exclusion list
            article_frame = pd.concat([article_frame, temp_article_frame],
                                      axis='index')  # append remaining articles to article_frame

        # remove duplicates and return:
        return article_frame.drop_duplicates(subset=id_column)

    @staticmethod
    def replace_letters(input_string: str, candidate_list: [str], replace_with=None,
                        omit_condition=(lambda x: False)) -> str:
        """
        Method that replaces or removes letters in a string.

        :param input_string: string with input text to be amended
        :param candidate_list: list of candidate characters
        :param replace_with: if None (default), the method just removes the candidate letters
        :param omit_condition: function to be checked that leaves a word unchanged if True (default is a function that always yields False)

        :return: amended text
        """
        # iterate over candidates:
        for candidate in candidate_list:
            if len(candidate) != 1: raise ValueError(
                f"Only characters (single letter strings) are allowed candidates. Problematic input: >> {candidate} <<")

        word_list = input_string.split()
        for ind, word in enumerate(word_list):
            # use translate function with uni-code representation (ord) to remove letters from everywhere
            word_list[ind] = word if omit_condition(word) else word.translate(
                {ord(i): replace_with for i in candidate_list})

        # join word list to string again:
        return " ".join(word_list)

    @staticmethod
    def replace_suffix(input_string: str, candidate_list: [str], replace_with="",
                       omit_condition=(lambda x: False), verbose=True, log_dict: dict = None) -> str:
        """
        Method that replaces or removes suffixes in a string.

        :param input_string: string with input text to be amended
        :param candidate_list: list of candidate characters or strings
        :param replace_with: if empty string (default), the method just removes the candidate suffixes
        :param omit_condition: function to be checked that leaves a word unchanged if True (default is a function that always yields False)
        :param verbose: if True (default), the method prints words that contain the candidates but not as suffix
        :param log_dict: dictionary to be updated during runtime {amended_word: [origin1, origin2, ...]}

        :return: amended text
        """
        # initialise word list:
        word_list = input_string.split()

        # iterate over candidates:
        for candidate in candidate_list:
            # iterate over word list and amend if word ends with candidate:
            for i, word in enumerate(word_list):
                # we do not use str.replace() to prevent errors when the candidate also occurs another time within the word:
                candidate_len = len(candidate)
                word_list[i] = word[:-candidate_len] + replace_with if (
                        word.endswith(candidate) and not omit_condition(word)) else word

                # save replaced word and result for later scrutiny if log_dict is provided:
                if log_dict is not None:
                    if word.endswith(candidate) and not omit_condition(word):  # if something was amended
                        if word_list[i] in log_dict.keys():  # if entry already present in dict
                            log_dict[word_list[i]].append(word)
                        else:
                            log_dict[word_list[i]] = [word]

                # eventually print status message:
                if verbose:
                    if not word.endswith(candidate) and candidate in word:
                        print(
                            f"Following word contains '{candidate}' but not at end, therefore remains unchanged: >> {word_list[i]} <<")

        # join word list to string again:
        return " ".join(word_list)

    @staticmethod
    def remove_numbers(input_string: str) -> str:
        """
        Remove words containing numbers from a string.

        This method filters out any word that starts or ends with a numeric character
        from the input string.

        Parameters
        ----------
        input_string : str
            The input string to process.

        Returns
        -------
        str
            A string with words containing numbers removed.
        """
        # detect numbers based on the first character:
        words_clean = [word for word in input_string.split() if (not word[0].isnumeric()) and (not word[-1].isnumeric())]
        # return cleaned list:
        return " ".join(words_clean)

    @staticmethod
    def replace_words(input_string: str, candidate_list: [str] = None, candidate_mapping: {str: str} = None) -> str:
        """
        Replace or remove specific words in a string.

        This method either removes words present in the candidate_list or replaces
        them based on candidate_mapping. Only one of these parameters can be provided
        at a time.

        Parameters
        ----------
        input_string : str
            The input string to process.
        candidate_list : list of str, optional
            A list of words to remove from the input string.
        candidate_mapping : dict of str to str, optional
            A mapping of words to their replacements.

        Returns
        -------
        str
            The processed string with words replaced or removed.

        Examples
        --------
        >>> replace_words("The quick brown fox", candidate_list=["quick"])
        'The brown fox'
        >>> replace_words("The quick brown fox", candidate_mapping={"quick": "slow"})
        'The slow brown fox'
        """
        assert (not (candidate_list is None and candidate_mapping is None)) and (not (
                    candidate_list is not None and candidate_mapping is not None)), "Either candidate_list (to remove candidates) or candidate_mapping (to replace candidates) needs to be provided, not both!"

        if candidate_list is not None:
            # create word_list only containing words not in candidate_list:
            word_list = [word for word in input_string.split() if word not in candidate_list]
        else:
            # create word_list only containing words not in candidate_mapping and mapped words from candidate_mapping:
            word_list = []
            for word in input_string.split():
                if word in candidate_mapping.keys():
                    word_list.append(candidate_mapping[word])
                else:
                    word_list.append(word)

        # reconvert list to string and return:
        return " ".join(word_list)

    @staticmethod
    def construct_abbreviation_dict(abbreviation_file, verbose=False) -> dict:
        """
        Create a dictionary of abbreviations and their meanings from a file.

        This method reads a file containing abbreviations and their meanings,
        returning a dictionary with lowercase keys and values. It handles
        duplicates and warns if overwriting existing entries.

        Parameters
        ----------
        abbreviation_file : str
            Path to the file containing abbreviation mappings.
        verbose : bool, optional
            If True, prints detailed processing information.

        Returns
        -------
        dict
            A dictionary of abbreviations and their meanings.

        Examples
        --------
        Given a file containing:
        ```
        abc -- abbreviation
        hpn -- home parenteral nutrition
        ```

        >>> construct_abbreviation_dict("abbreviation_file.txt")
        {'abc': 'abbreviation', 'hpn': 'home_parenteral_nutrition'}
        """
        # read abbreviation list:
        abbreviation_mapping = dict()
        with open(abbreviation_file) as f:
            for line in f.readlines():
                if line[0] == "#": continue  # skip comments
                if line.strip() == "": continue  # skip empty lines
                # read abbreviations and meanings:
                try:
                    abbreviation, meaning = line.split("--")
                except ValueError:
                    continue

                # remove leading and trailing spaces and add to mapping:
                abbreviation = abbreviation.strip().lower()
                meaning = meaning.strip().lower()

                # add underscores for latter replacement in text:
                if len(meaning.split()) != 1:
                    meaning = "_".join(meaning.split())

                # add to dict and warn if already present:
                if abbreviation in abbreviation_mapping.keys():
                    if meaning == abbreviation_mapping[abbreviation]: continue
                    print(f"[WARNING] {abbreviation} was found multiple times. Previous meaning ({abbreviation_mapping[abbreviation]}) is now overwritten with {meaning}!")
                abbreviation_mapping.update({abbreviation: meaning})

                # explanatory statement:
                if verbose: print(f"Found {abbreviation} with meaning: {meaning}.")

        return abbreviation_mapping

    @staticmethod
    def detect_n_grams(input_text_list: [str], n=2, occurrence_threshold=50) -> {str: int}:
        """
        Detect n-grams from a list of input texts and filter them by occurrence.

        This method generates n-gram candidates, counts their occurrences,
        and retains only those above the specified threshold.

        Parameters
        ----------
        input_text_list : list of str
            A list of input texts to analyze.
        n : int, optional
            The size of n-grams to generate (default is 2).
        occurrence_threshold : int, optional
            The minimum number of occurrences for an n-gram to be retained.

        Returns
        -------
        dict
            A dictionary of n-grams and their occurrences.

        Examples
        --------
        >>> detect_n_grams(["hello world", "world of code"], n=2, occurrence_threshold=1)
        {'hello_world': 1, 'world_of': 1, 'of_code': 1}
        """
        # Download necessary NLTK resources
        nltk.download('punkt', quiet=True)

        # construct list with all possible n-grams:
        possible_n_grams = list()
        print('Creating n-gram candidates...')
        for input_text in tqdm(input_text_list):  # iterate over input texts
            tokens = word_tokenize(input_text)  # tokenization
            possible_n_grams += list(ngrams(tokens, n))

        # count occurrences:
        print('Counting occurrences...')
        occurrences = Counter(possible_n_grams)  # very (most) efficient way of counting

        # now merge tuples to string representation and enforce occurrence_threshold:
        remaining_n_grams = {"_".join(n_gram): occurrences[n_gram] for n_gram in occurrences.keys() if occurrences[n_gram] > occurrence_threshold}
        print(f"Based on an occurrence threshold of {occurrence_threshold}, {len(remaining_n_grams)} of possible {len(possible_n_grams)} n-grams have been kept.")

        return remaining_n_grams

    @staticmethod
    def connect_compound_words(input_string: str, candidate_list: [str]) -> str:
        """
        Connect compound words in a string using underscores.

        This method replaces compound words from the candidate list with their
        underscore-connected versions, if found in the input string.

        Parameters
        ----------
        input_string : str
            The input text to process.
        candidate_list : list of str
            A list of compound word strings to replace with their underscored versions.

        Returns
        -------
        str
            The processed string with compound words connected.

        Examples
        --------
        >>> connect_compound_words("The patient needs home parenteral nutrition.", ["home parenteral nutrition"])
        'The patient needs home_parenteral_nutrition.'
        """
        # initialise lists:
        output_word_list = []
        input_word_list = input_string.split()

        # skip single word candidates and prepare check list: (while replacing underscores)
        candidate_tuple_list = [tuple(candidate.replace("_", " ").split()) for candidate in candidate_list if len(candidate.replace("_", " ").split()) > 1]
        check_list = [candidate[0] for candidate in candidate_tuple_list]

        # iterate over input and check for candidates:
        skip_counter = 0  # we need to sometimes skip multiple iterations, which is achieved through this variable
        for word_ind, word in enumerate(input_word_list):
            if skip_counter > 0: skip_counter -= 1; continue  # skip iteration if skip counter is above 0 and reduce such by one

            # check whether a candidate starts with word:
            if word in check_list:
                # check whether the other compound parts (candidate) are present in the subsequent words from input_word_list:
                candidate = candidate_tuple_list[check_list.index(word)]
                if candidate == tuple(input_word_list[word_ind:word_ind + len(candidate)]):
                    output_word_list.append("_".join(candidate))  # append the compound word indicated by underscores
                    skip_counter = len(candidate) - 1;
                    continue  # this achieves skipping to the first word after all compound parts

            # if any of the if-checks failed, the word is added without amendment
            output_word_list.append(word)

        # reconvert list to string and return:
        return " ".join(output_word_list)

    @staticmethod
    def remove_words_below_tolerance_from_string(input_string: str, tolerance: float, verbose=True) -> str:
        """
        Remove words occurring less frequently than a tolerance ratio.

        This method removes words that occur less than the specified
        tolerance ratio relative to the highest-frequency word.

        Parameters
        ----------
        input_string : str
            The input string to process.
        tolerance : float
            The minimum occurrence ratio (0.0 to 1.0) for words to be retained.
        verbose : bool, optional
            If True, prints details of the words removed.

        Returns
        -------
        str
            The processed string with less frequent words removed.

        Examples
        --------
        >>> remove_words_below_tolerance_from_string("apple apple orange banana", 0.5)
        'apple apple'
        """
        # count occurrences of each word:
        word_list = input_string.split()
        # utilize set to avoid duplicates:
        count_dict = {word: word_list.count(word) for word in set(input_string.split())}

        # calculate ratio:
        highest_count = max(count_dict.values())
        ratios = {word: count / highest_count for word, count in count_dict.items()}

        # derive highest-frequency word, eventually insightful to add as stop word:
        highest_freq_word = "ERROR: WORD NOT FOUND!"
        for word, count in count_dict.items():
            if count == highest_count:
                highest_freq_word = word
                break  # stop iterating

        # retain only values with ratio above tolerance:
        words_clean = [word for word in word_list if ratios[word] >= tolerance]
        removed_words = [word for word in word_list if ratios[word] < tolerance]

        if verbose:
            if len(removed_words) > 0:
                print(
                    f"Removed {len(removed_words)} words because they occurred less than {tolerance * 100}% compared to highest frequency word ('{highest_freq_word}').\nWords removed: {removed_words}")

        # rejoin and return:
        return " ".join(words_clean)


############### auxiliary methods ###############
def assess_abbreviations(candidates: dict, abbreviation_file) -> None:
    """ Manual assessment of abbreviations from candidate dictionary """
    current_abbreviations = TextCorpusPreprocessor.construct_abbreviation_dict(abbreviation_file)  # load already included ones
    for counter, (abbreviation, meaning) in enumerate(candidates.items()):
        if abbreviation in current_abbreviations.keys(): continue  # avoid redundancy

        # ask for user input to save abbreviation in list:
        answer = input(
            f"No. {counter}/{len(candidates)}\nPlease see the following abbreviation and meaning. If both are fine, press CMD+Enter (provide an empty message), if you want to change the meaning, enter the correct one, if you want to ignore the abbreviation, enter 'n' or 'no'.\n You can also enter a number (1 or 2), to indicate that the meaning is correct, but should start after that word.\n\n{abbreviation} - {meaning}")
        if answer.lower() == "no" or answer.lower() == "n":  # skip
            continue
        elif answer == "1" or answer == "2":
            meaning = " ".join(meaning.split()[int(answer):])
        elif answer != "":  # amend meaning
            meaning = answer
        # attach abbreviation and meaning to abbreviation list:
        with open(abbreviation_file, 'a') as file:
            file.write(f"{abbreviation} -- {meaning}\n")


def detect_abbreviations(input_string: str, abbreviation_length_threshold=5) -> {str: str}:
    """ Detecting abbreviations from input text and returning a mapping dictionary (abbreviation -> meaning) """
    # these are characters that are forbidden within abbreviations (mostly punctuation)
    character_black_list = [".", "?", "!", ",", "_", "[", "]", "(", ")", "/", "\"", "'", "+", "<", ">", "%", "•",
                            "º",
                            "±", "~", "$", "…", "=", "*"]
    input_word_list = input_string.split()
    possible_abbreviations = {}

    for word_ind, word in enumerate(input_word_list):
        # upon first usage abbreviations are very commonly introduced in brackets
        # we further check that there is more than one letter between these brackets:
        if word[0] == "(" and word[-1] == ")":
            abbreviation = word[1:-1]
            abbreviation_length = len(abbreviation)
            if 1 < abbreviation_length <= word_ind:  # the comparison to word_ind prevents indexing errors
                # further we check that 1) the length is within the pre-defined threshold, 2) that it contains no punctuation marks, i.e. character_black_list and 3) that it contains not only numbers:
                if abbreviation_length > abbreviation_length_threshold:  continue
                if True in [(forbidden_character in abbreviation) for forbidden_character in
                            character_black_list]: continue  # utilize a boolean list for condition 2
                if not False in [char.isnumeric() for char in
                                 abbreviation]: continue  # again boolean list for condition 3

                # assume that the previous words indicate the meaning:
                possible_meaning = " ".join(input_word_list[word_ind - abbreviation_length: word_ind])

                # add abbreviation and meaning all in lower case to dictionary:
                possible_abbreviations[abbreviation.lower()] = possible_meaning.lower()

    return possible_abbreviations


def assess_abbreviations_prompt(candidates: dict, abbreviation_file, long_prompt=True, limit_length=None) -> str:
    """ Semi-automatic assessment of abbreviations using LLMs """
    # load and skip already included ones:
    current_abbreviations = TextCorpusPreprocessor.construct_abbreviation_dict(abbreviation_file, verbose=False)
    remaining_abbreviations = {abb: mean for abb, mean in candidates.items() if
                               abb not in current_abbreviations.keys()}

    # initialise prompt parts:
    abbreviation_string = ""
    for length, (abbr, meaning) in enumerate(remaining_abbreviations.items()):
        if limit_length is not None and length > limit_length: break  # this helps not to exceed context length
        abbreviation_string += abbr + " -- " + meaning + "\n"
    pre_prompt = "You are a linguist specialized on medical and technical abbreviations. Please review the following abbreviations with their meanings and remove misleading (meaning those, that can be only recognized from context) or meaningless abbreviations, as well as change wrong meanings.\n"
    examples = """\nTwo examples of good solutions: \nai -- artificial intelligence\necg -- electrocardiogram\n\n"""
    question = f"Now please optimise the following abbreviation candidates. Please answer only with your optimised solution in the shown format (abbreviation -- meaning):\n" + abbreviation_string

    # join parts:
    prompt = pre_prompt + examples + question if long_prompt else pre_prompt + question
    return prompt