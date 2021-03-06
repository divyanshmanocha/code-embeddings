"""
## App: NLP App with Streamlit (NLPiffy)
Author: [Jesse E.Agbe(JCharis)](https://github.com/Jcharis))\n
Source: [Github](https://github.com/Jcharis/Streamlit_DataScience_Apps/)
Credits: Streamlit Team,Marc Skov Madsen(For Awesome-streamlit gallery)
Description
This is a Natural Language Processing(NLP) Based App useful for basic NLP concepts such as follows;
+ Tokenization & Lemmatization using Spacy
+ Named Entity Recognition(NER) using SpaCy
+ Sentiment Analysis using TextBlob
+ Document/Text Summarization using Gensim/Sumy
This is built with Streamlit Framework, an awesome framework for building ML and NLP tools.
Purpose
To perform basic and useful NLP task with Streamlit,Spacy,Textblob and Gensim/Sumy
"""
# Core Pkgs
import streamlit as st 
import os
import streamlit.components.v1 as components
import pandas as pd
# NLP Pkgs
from model import ExactMatches, BaselineModel, SourceCodeModel, PseudoJointEmbedding, cluster_embeddings
from model import SINGLE_WORD_1, SINGLE_WORD_2, SINGLE_WORD_3, SINGLE_WORD_4
from model import SENTENCE_1, SENTENCE_2, SENTENCE_3, SENTENCE_4
from model import CODE_1, CODE_2, CODE_3, CODE_4
from model import CODE_DESC_1, CODE_DESC_2, CODE_DESC_3, CODE_DESC_4

DATAFRAME_PATH = 'test_df.pkl'


def clustering_page():
    st.write("We can perform an analysis of topics using Latent Dirichlet Allocation (LDA) at a very basic level, or on the embedding itself.")
    st.write("Query a basic eucledian clustering in the embedding space below.")
    st.markdown("**Purpose**: We can use this to infer file structures")
    max_num = st.slider("Number of clusters to use", 3, 20, 18)
    disp_num = st.slider("Select a cluster to display", 0, max_num-1, 1)
    if st.button("Show results"):
        df = pd.read_pickle(DATAFRAME_PATH)
        result = cluster_embeddings(df, max_num, disp_num)
        st.dataframe(result)
    


def eda_page():
    st.write("The dataset was built by parsing all files using Python's native Abstract Syntax Tree (AST) parser at the lowest possible level. Information at a method level was then extracted.")
    st.write("As a bonus this therefore enabled me to visualise the entire structure of any project directory by analysing dependencies (see below an undirected graph)")
    HtmlFile = open("vis.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    components.html(source_code, height=600)
    st.write("The thickness of the lines indicate the dependency of the methods (proportional to number of calls). Insignificant methods such as print were removed for cleaner visualisation.")


def show_run_query(model, examples, id_):
    message = st.text_area("Enter text",examples[0])
    num = st.slider('How many results would you like to see?', 1, 5, 3, key=id_*2)
    if st.button("Analyze", key=id_):
        df = pd.read_pickle(DATAFRAME_PATH)
        results = message
        m = model(df)
        if isinstance(m, ExactMatches):
            matches = m.exactWordMatches(message)
            results = m.firstNMatches(matches, num)
        else:
            results = m.match_sentence(message, num)
            results = results[0]
        st.write(results)
    code = '\n'.join([str(x) for x in examples])
    st.code(code, language='python')


def nmt_page():
    st.markdown("**Work in progress**")
    st.write("The purpose of this is to be able to generate natural language descriptions from source code. This is useful for projects as well as future machine learning models for data augmentation.")


def home_page():
    id_ = 100
    st.markdown("""The purpose of this quickly-put-together app is to illustrate the usefulness of embeddings, even for small datasets""")
    st.write("A pseudo joint embedding model that has been built primarily using Google's Universal Sentence Encoder")
    st.subheader('Purpose')
    st.write('Primarily removing code duplication, and therefore saving developer time as well as avoiding re-introduction of bugs. Imagine suggestions of similar code in the system as you are coding.')
    st.markdown('**The page walks through all results of the project. The backend is running a lite version of the embedding model.**')
    
    st.write('As an example lets try to find the following code fragments (that we know exist) from the mlfromscratch datasets:')
    code = '\n'.join([str(x) for x in [SENTENCE_1, SENTENCE_2, SENTENCE_3, SENTENCE_4]]) 
    st.code(code, language='python')
    
    st.subheader("Current Method: exact keyword match")
    st.markdown("**Try querying (the search includes comments and docstrings!)**")
    st.write("We have to manually pick keywords that we think will be unique. Examples shown below.")
    show_run_query(ExactMatches, [SINGLE_WORD_1, SINGLE_WORD_2, SINGLE_WORD_3, SINGLE_WORD_4], id_)
    
    id_ += 1
    
    st.subheader("Descriptions Embeddings: semantic sentence match")
    st.markdown("**Try querying using sentences**")
    st.write("Examples shown below. None of the examples are exact matches.")
    show_run_query(BaselineModel, [SENTENCE_1, SENTENCE_2, SENTENCE_3, SENTENCE_4], id_)

    id_ += 1
    
    st.subheader("Code Embeddings: semantic code match")
    st.markdown("**Try querying using similar code**")
    st.write("Examples shown below. None of the examples are exact matches or use the same variables.")
    show_run_query(SourceCodeModel, [CODE_1, CODE_2, CODE_3, CODE_4], id_)

    id_ += 1
    
    st.subheader("Pseudo joint Embeddings: semantic code and descriptions match")
    st.markdown("**Try querying using incomplete code and descriptions**")
    st.write("Examples shown below.")
    show_run_query(PseudoJointEmbedding, [CODE_DESC_1, CODE_DESC_2, CODE_DESC_3, CODE_DESC_4], id_)


def main():
    """ NLP Based App with Streamlit """

	# Title
    st.header("Joint Embeddings for code and descriptions")


    st.sidebar.subheader("Pseudo joint embeddings")
    st.sidebar.info("dataset: mlfromscratch")
    st.sidebar.write("This is a proof of concept of a richer idea on a very small but real-world dataset running on a lite model")
	
    page = st.sidebar.selectbox("Current research results", ["Search using embeddings", "Exploring and parsing the dataset", "Clustering using embeddings", "Neural Machine Translation"])

    if page == "Exploring and parsing the dataset":
        eda_page()
    
    elif page == "Clustering using embeddings":
        clustering_page()

    elif page == "Neural Machine Translation":
        nmt_page()

    else:
        home_page()

    st.sidebar.subheader("By")
    st.sidebar.text("Divyansh Manocha")
	

if __name__ == '__main__':
	main()
