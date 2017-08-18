<p align="center">
<img width=50% src="./img/logo_docluster.png">
</p>

<p align="center">
<img width=20% src="./img/title.png">
</p>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![Pypi](https://img.shields.io/pypi/v/docluster.svg)](https://pypi.python.org/pypi/docluster)
[![Travis-Cli](https://img.shields.io/travis/metinsay/docluster.svg)](https://travis-ci.org/metinsay/docluster)
[![GitHub Issues](https://img.shields.io/github/issues/metinsay/docluster.svg)](https://github.com/metinsay/docluster/issues)
[![Documentation](https://readthedocs.org/projects/docluster/badge/?version=latest)](https://docluster.re˜t/?badge=latest)
[![Pyup](https://pyup.io/repos/github/metinsay/docluster/shield.svg)](https://pyup.io/repos/github/metinsay/docluster/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)




**docluster** is an open source project that aims to bring the natural language processing community together. The demand for NLP increases each year as processing power increases and machine learning advances. As a result, many research projects are conducted on NLP and related subjects. Keeping up with these advancements from all around the world is nearly impossible. docluster tries to bring all studies from all countries into a library that can be easily applied, visualized and deployed.

**You want to contribute? Great! See [Contribute](./CONTRIBUTING.md) section.**

## Table of Contents

- [Vision](#vision)
- [Upcoming Features](#upcoming-features)
    + [Most recent research models](#most-recent-research-models)
    + [Docluster Package Manager (dpm)](#docluster-package-manager--dpm-)
- [Installation](#installation)
- [Contribute](#contribute)
- [Release History](#release-history)
- [Maintainers](#maintainers)
- [Puzzle](#puzzle)

## Vision

Natural language processing space is populated with many individual and organizational projects that provide specific needs of consumer. Unfortunately none so far tries to serve future innovation. docluster aims to fill this void by being:

**Free:** Many projects that provide powerful NLP tools come with a price attach to it (usually a monthly subscription or a per request). Even seeing the 'Pricing' tab on the website of these projects, discourages people who are genuinely interested to contribute. docluster aims to break this wall by providing these tools and many more with open source code.

**Up-to-date on research:** There are so many research projects conducted that it is almost impossible to keep up. These projects are usually presented through a white paper and occasionally a repo separately created just for that particular research. docluster aims to keep people up-to-date by bringing tools in a singular repo.

**Open to contributions:** Large organizations such as Google, Facebook, MIT are the leading most of the research projects in NLP. However, there are many individuals who are making innovative contributions to the community. docluster aims to be a platform for these individuals. All the credit still stay with the individual that invented the new model.

**All-language support:** Although language is a major factor in NLP, most efforts are put into more common languages like English. However, there are research projects in other languages as well and these projects are usually conducted in their respective country. docluster tries to join all these research projects by inviting people from all around the world to contribute.

**Easy to use:** Many NLP models require some sort of training step before any usage. docluster gives the user the option to train their models or use a pre-trained model. It comes with a pre-trained model catalog where users can programmatically download or upload pre-trained model.

## Upcoming Features

#### Most recent research models



```python
ft = FastText()
embeddings = ft.fit(docs)
```

#### Docluster Package Manager (dpm)

**dpm** is a package manager that removes tedious steps of NLP and enables researchers and software engineers to focus on their own work. dpm provides access to a catalog of models implemented/trained by fellow NLP community members and a catalog of corpora.

##### A Catalog of Pre-trained/Pre-implemented Models
Models can be pre-trained models like Word2Vec, Sentiment Analysis, Chatbots or just the skeleton code of a model.</br></br>
**Note:** dpm **does not store the Python code** but only the Python object by using pickling. The person uploading the model can decide to attach their source code as a reference. </br></br>

```python
# Word2Vec pre-trained with English Wikipedia
word2vec = dpm.download_model('docluster/word2vec_en_wikipedia').model

# Uploading a chatbot
dpm.upload(chatbot, model_name='chatbot_slack_tr', accuracy=90.6
                    description='A Slack Chatbot that can communicate in Turkish.',
                    user_name='metinsay', token='A32CD1G3J5D6DE',
                    citation='github.com/metinsay', tags=['slack', 'chatbot', 'turkish'])
```
#### Corpus Catalog


    ```python
    # Random half of English Wikipedia documents
    wikis = dpm.download_corpus('docluster/wikipedia_en', shuffle=True, fraction=0.5).corpus

    # Random quarter of the combined set of English Turkish Wikipedia and Google News, CNN articles. Also manually limits the total size to 1 GB
    wikis = dpm.download_corpora(['docluster/wikipedia_en','docluster/googlenews_en','docluster/cnn_en'], shuffle=True, fraction=0.25, limit_size=1000).corpus
    ```

* **Flow** - A powerful pipelining tool with branching. Underlying graph can consist of unconnected subgraphs.

    ```python
    pre = Preprocessor()
    tfidf = TfIdf(min_df=0.2, max_df=0.5)
    km = BisectingKMeans(2, dist_metric=DistanceMetric.manhattan)
    db = DBScan()

    # pre - tfidf - kmeans
    #            \
    #             db

    flo = Flow(do_thread_branches=True)
    flo.chain(pre, tfidf, km)
    flo.link(tfidf, db)
    km_clusters, db_clusters = flo.fit(docs)
    ```


* Built-in progress visualization

    ```python
    # Plots a voronoi diagram that adapts to new cluster assignments on each epoch
    km = Kmeans(4, plot_progress=True, plot_result=True)
    clusters = km.fit(docs)
    ```

## Installation

**The project is under heavy construction currently.**

## Contribute

We really expect this project to be the project of the community. Therefore, we welcome EVERY contribution from everyone that are interested in NLP or related areas. Please check out [how to contribute](./CONTRIBUTING.md). If you have any questions after reading the documents, contact the admins of the project.

## Release History

Project is still in development.

## Maintainers

**Metin Say -** MIT '19. metin@mit.edu

See a list of all contributers at [Authors](AUTHORS.md).

## Puzzle

**Hint 1:** Do not ignore me!

(We recommend cloning the repo before starting the puzzle.)
