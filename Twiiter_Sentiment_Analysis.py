
from csv import DictReader
import pickle
import re, math, collections, itertools
from nltk.corpus import stopwords
import string

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

import nltk, nltk.classify.util, nltk.metrics
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import time

def readDataset(csv_file):
    pos_tweets=[]
    neg_tweets=[]
    senti=[]
    tweets_samples=[]
    #csv_file='Sentiment Analysis Dataset.csv'
    with open(csv_file) as f:
        for row in DictReader(f):
            label= int(row["Sentiment"])
            senti.append(label)
            tweets_samples.append(row["SentimentText"])
            if label ==0:
                neg_tweets.append(row["SentimentText"])
            else:
                pos_tweets.append(row["SentimentText"])
    #print pos_tweets
    #f = open('postweets.pickle', 'wb')
    #pickle.dump(pos_tweets, f)
    #f.close()

    #f = open('negtweets.pickle', 'wb')
    #pickle.dump(neg_tweets, f)
    #f.close()
    return pos_tweets,neg_tweets
#end of readDataset

#start process_tweet
def processTweet(tweet):
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end processTweet

#start getStopWordList
def getStopWordList():
    punctuation = list(string.punctuation)
    stopWords= stopwords.words('english') + punctuation + ['AT_USER','URL']
    return stopWords
#end getStopWords

#start getfeatureVector
def getFeatureVector(tweet):
    featureVector = []
    #split tweet into words
    words = tweet.split()
    # print words
    for w in words:
        #1 - replace two or more with two occurrences
        pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
        w= pattern.sub(r"\1\1", w)

        #2- strip punctuation
        w = w.strip('\'"?,.')

        #3- check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)

	#4- ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector
#end getfeatureVector

#start words_Scores
def create_word_scores(pos_tweets,neg_tweets):
    #creates lists of all positive and negative words
    posWords = []
    negWords = []
    for twe in neg_tweets:
        processedTweet = processTweet(twe)
        negWords.append(getFeatureVector(processedTweet))

    for twe in pos_tweets:
        processedTweet = processTweet(twe)
        posWords.append(getFeatureVector(processedTweet))

    posWords = list(itertools.chain(*posWords))
    negWords = list(itertools.chain(*negWords))

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()

    for word in posWords:
	w= word.lower()
        word_fd[w]=word_fd[w]+1;
        cond_word_fd['pos'][w]=cond_word_fd['pos'][w]+1
    for word in negWords:
        w=word.lower()
        word_fd[w]= word_fd[w]+1
        cond_word_fd['neg'][w]= cond_word_fd['neg'][w]+1

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
    return word_scores
#end create_wordScores


#start featureSelection
def feature_Selection(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words
# end Feature_Selection


def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])

# start feature_Extraction
def feature_Extraction(feature_select,pos_tweets,neg_tweets):
    #f = open('postweets.pickle', 'rb')
    #pos_tweets= pickle.load(f)
    #f.close()

    #f = open('negtweets.pickle', 'rb')
    #neg_tweets= pickle.load(f)
    #f.close()

    negfeats=[]
    posfeats=[]
    for twe in neg_tweets:
        processedTweet = processTweet(twe)
        negfeats.append((feature_select(getFeatureVector(processedTweet)),'neg'))

    for twe in pos_tweets:
        processedTweet = processTweet(twe)
        posfeats.append((feature_select(getFeatureVector(processedTweet)),'pos'))
    return posfeats, negfeats
#end feature_Extraction

# using all the extraction festures without optimizaton
#def make_full_dict(words):
#    return dict([(word, True) for word in words])
#print 'using all words as features'
#evaluate_features(make_full_dict)

#start Train CLassifier
def train_Classifier(posfeats,negfeats,index):
    # divide dataset into train and validation sets
    posCutoff = int(math.floor(len(posfeats)*7/10))
    negCutoff = int(math.floor(len(negfeats)*7/10))
    trainFeatures = posfeats[:posCutoff] + negfeats[:negCutoff]
    testFeatures = posfeats[posCutoff:] + negfeats[negCutoff:]

    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)

    classsifiername=''

    if (index == 0):
        classifier = nltk.classify.maxent.MaxentClassifier.train(trainFeatures, 'GIS', trace=3, encoding=None, labels=None, gaussian_prior_sigma=0, max_iter = 5)
        classsifiername= 'Maximum Entropy'
    elif (index ==1):
        classifier = SklearnClassifier(BernoulliNB())
	classifier.train(trainFeatures)
        classsifiername='Bernoulli Naive Bayes'
    else:
        classifier = SklearnClassifier(LogisticRegression())
	classifier.train(trainFeatures)
        classsifiername = 'LogisticRegression'

    for i, (features, label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)

    print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
    print 'pos precision:', nltk.metrics.precision(referenceSets['pos'], testSets['pos'])
    print 'pos recall:', nltk.metrics.recall(referenceSets['pos'], testSets['pos'])
    print 'neg precision:', nltk.metrics.precision(referenceSets['neg'], testSets['neg'])
    print 'neg recall:', nltk.metrics.recall(referenceSets['neg'], testSets['neg'])
    #classifier.show_most_informative_features(10)
    return classifier
#end trainClassifier


start = time.clock()

print 'Loading Tweets Dataset'
pos_tweets,neg_tweets = readDataset('Sentiment Analysis Dataset.csv')

stopWords = getStopWordList()
word_scores= create_word_scores(pos_tweets,neg_tweets)

num = 30000
print 'evaluating best %d word features' % (num)
best_words = feature_Selection(word_scores, num)
posfeats, negfeats = feature_Extraction(best_word_features,pos_tweets,neg_tweets)

classifier_names = [ 'MaximumEntropy_classifier','BernoulliNB_classifier','LogisticRegression_classifier']
index=0
for name in classifier_names:
    classifier= train_Classifier(posfeats,negfeats,index)
    index=index+1
    f = open(name+'.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()

endT = time.clock()
elapsed = endT - start

print "Time spent in Twitter Sentiment Analysis (Preprocessing + Training 3 classifiers) is: ", elapsed


