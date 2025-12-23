import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cityblock, euclidean
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz
from scipy.sparse import hstack


nltk.download("punkt")
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')


def common_words(q1, q2):
    return len(set(q1) & set(q2))

def jaccard(q1, q2):
    if len(set(q1) | set(q2)) == 0:
        return 0
    return len(set(q1) & set(q2)) / len(set(q1) | set(q2))

def ngram_overlap_tokens(t1, t2, n=2):
    ngrams1 = set(tuple(t1[i:i+n]) for i in range(len(t1)-n+1))
    ngrams2 = set(tuple(t2[i:i+n]) for i in range(len(t2)-n+1))

    if len(ngrams1 | ngrams2) == 0:
        return 0

    return len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN 

def lemma_tokenizer(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]

    pos_tags = nltk.pos_tag(tokens)
    lemmas = [
        lemmatizer.lemmatize(token, get_wordnet_pos(tag))
        for token, tag in pos_tags
    ]
    return lemmas

def data_processing(df):

    df['q1_tokens'] = df['question1'].apply(lemma_tokenizer)
    df['q2_tokens'] = df['question2'].apply(lemma_tokenizer)
    
    df['q1_clean'] = df['q1_tokens'].apply(lambda x: " ".join(x))
    df['q2_clean'] = df['q2_tokens'].apply(lambda x: " ".join(x))
    
    vectorizer = TfidfVectorizer()
    vectorizer.fit(pd.concat([df["q1_clean"], df["q2_clean"]]))
    tfidf_q1 = vectorizer.transform(df["q1_clean"])
    tfidf_q2 = vectorizer.transform(df["q2_clean"])
    
    df["common_words"] = df.apply(lambda x: common_words(x['q1_tokens'], x['q2_tokens']), axis=1)
    df["jaccard"] = df.apply(lambda x: jaccard(x['q1_tokens'], x['q2_tokens']), axis=1)
    df['lev_ratio'] = df.apply(lambda x: fuzz.ratio(str(x['question1']), str(x['question2'])) / 100, axis=1)
    df["bigram_overlap"] = df.apply(lambda row: ngram_overlap_tokens(row.q1_tokens, row.q2_tokens, n=2),axis=1)
    df["trigram_overlap"] = df.apply(lambda row: ngram_overlap_tokens(row.q1_tokens, row.q2_tokens, n=3),axis=1)

    return df

def tfidf_features(df, vectorizer=None):
    df = data_processing(df)
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        vectorizer.fit(pd.concat([df["q1_clean"], df["q2_clean"]]))


    tfidf_q1 = vectorizer.transform(df["q1_clean"])
    tfidf_q2 = vectorizer.transform(df["q2_clean"])

    # distance features
    df["cosine_sim"] = [cosine_similarity(tfidf_q1[i], tfidf_q2[i])[0][0] 
                        for i in range(len(df))]
    df["manhattan"] = [cityblock(tfidf_q1[i].toarray().ravel(),
                                 tfidf_q2[i].toarray().ravel()) 
                       for i in range(len(df))]
    df["euclidean"] = [euclidean(tfidf_q1[i].toarray().ravel(),
                                 tfidf_q2[i].toarray().ravel()) 
                       for i in range(len(df))]

    return df, vectorizer, tfidf_q1, tfidf_q2

def deploy_features(df, vectorizer):
    df = data_processing(df)
    train_cols = ["common_words",	"jaccard", "lev_ratio",	"bigram_overlap",	"trigram_overlap",	"cosine_sim",	"manhattan", "euclidean", "sbert_cosine"]

    tfidf_q1 = vectorizer.transform(df["q1_clean"])
    tfidf_q2 = vectorizer.transform(df["q2_clean"])

    # distance features
    df["cosine_sim"] = [cosine_similarity(tfidf_q1[i], tfidf_q2[i])[0][0] 
                        for i in range(len(df))]
    df["manhattan"] = [cityblock(tfidf_q1[i].toarray().ravel(),
                                 tfidf_q2[i].toarray().ravel()) 
                       for i in range(len(df))]
    df["euclidean"] = [euclidean(tfidf_q1[i].toarray().ravel(),
                                 tfidf_q2[i].toarray().ravel()) 
                       for i in range(len(df))]
    
    tfidf = hstack([tfidf_q1, tfidf_q2])
    X_stack = hstack([tfidf, df[train_cols].values])
    return X_stack