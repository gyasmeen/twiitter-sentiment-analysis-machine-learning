# run the file for training
#python Twitter_Sentiment_Analysis.py --training_flag=1

# run the file for testing
#python Twitter_Sentiment_Analysis.py --training_flag=0 --couchdb_ip=115.146.95.99:5984 --db="tweets"




import json
from couchdb import Server

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


#start Generate Classifiers

def getbest_words():
    print 'Loading Tweets Dataset'
    pos_tweets,neg_tweets = readDataset('Sentiment Analysis Dataset.csv')

    word_scores= create_word_scores(pos_tweets,neg_tweets)

    num = 25000
    print 'evaluating best %d word features' % (num)
    best_words = feature_Selection(word_scores, num)

    return best_words, pos_tweets,neg_tweets

#end

def generate_classifiers( pos_tweets,neg_tweets):

    start = time.clock()
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
#end generate classifiers

#start
def get_tweet_senti(tw):

    #print('%s -- %s\n' % (tw['user']['screen_name'], tw['text']))
    processedTweet = processTweet(tw['text'])
    FV=best_word_features(getFeatureVector(processedTweet))

    dist=ME_classifier.prob_classify(FV)
    ME_prob_pos = dist.prob("pos")
    ME_prob_neg = dist.prob("neg")

    dist=BerNB_classifier.prob_classify(FV)
    BerNB_prob_pos = dist.prob("pos")
    BerNB_prob_neg = dist.prob("neg")

    dist=LR_classifier.prob_classify(FV)
    LR_prob_pos = dist.prob("pos")
    LR_prob_neg = dist.prob("neg")

    prob_pos = ( ME_prob_pos+ BerNB_prob_pos+ LR_prob_pos)/3
    prob_neg = ( ME_prob_neg+ BerNB_prob_neg+ LR_prob_neg)/3

    if(prob_pos > 0.7):
	senti = 'positive'
    elif (prob_neg > 0.7):
	senti = 'negative'
    else:
	senti = 'neutral'

    #if (tw['geo'] ==null):
    senti_score ={'tweet-features': FV ,'sentiment':senti, 'positive-perc':prob_pos, 'negative-perc':prob_neg}
    #else :
#	loc_point=tw['geo']
#   	g = geocoder.google(loc_point['coordinates'], method='reverse')
#   	modified_place={'geo_city': g.city,'geo_state': g.state_long ,'geo_country':g.country_long,'geo_bbox': g.bbox})
#	senti_score ={'tweet-features': FV ,'sentiment':senti, 'positive-perc':prob_pos, 'negative-perc':prob_neg,'place-mod': modified_place}


    new_tw = tw.copy()
    new_tw.update(senti_score)

    #db.save(new_tw)
    return new_tw
#end


import urllib2
import zipfile

def download_sentiment_training_dataset():
    url = "http://thinknook.com/wp-content/uploads/2012/09/Sentiment-Analysis-Dataset.zip"

    file_name = url.split('/')[-1]
    u = urllib2.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (file_name, file_size)

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,

    f.close()
    print "Downloading: Dataset File: %s" %file_name
    zip_ref = zipfile.ZipFile(file_name, 'r')
    zip_ref.extractall()
    zip_ref.close()

def update_couchdb():
#    for doc in db.view('sentiment-analysis/count_tweets'):
#        print doc.value
    count=1;
    for doc in  db.view('sentiment-analysis/get_tweets'):
        tw=doc.value
        new_tw= get_tweet_senti(tw)
        new_tw.update({'_id': tw['id_str']})
        try:
            db.save(new_tw)
        except:
            print ("Tweet " + tw['id_str'] + " already exists !!!! " )
        #print doc #doc['_id'], doc['_rev']
        print count
        count=count+1
        max_id=new_tw['id']
    return max_id

def _create_views(self):
    count_map = 'function(doc) { emit(doc.id, 1); }'
    count_reduce = 'function(keys, values) { return sum(values); }'
    view = couchdb.design.ViewDefinition('twitter','count_tweets',count_map,reduce_fun=count_reduce)
    view.sync(self.db)
    get_tweets = 'function(doc) {    if (doc.id)    {	  emit(doc.id, doc);    }}'
    view = couchdb.design.ViewDefinition('sentiment-analysis', 'get_tweets', get_tweets)
    view.sync(self.db)






import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--training_flag', '-tr', type=int, help='1 for traing and generating classifiers else for testing')
parser.add_argument('--couchdb_ip', '-ip', help='ip of couchdb')
parser.add_argument('--dbname', '-db', help='db name')
args = parser.parse_args()

stopWords = getStopWordList()

#best_words,pos_tweets,neg_tweets=getbest_words()

#f = open('bestwords.pickle', 'wb')
#pickle.dump(best_words, f)
#f.close()
f = open('bestwords.pickle', 'rb')
best_words = pickle.load(f)
f.close()

trainflag= args.training_flag

if (trainflag==1):
    download_sentiment_training_dataset()
    generate_classifiers( pos_tweets,neg_tweets)
else:
    print 'loading classifiers'
    f = open('MaximumEntropy_classifier.pickle', 'rb')
    ME_classifier = pickle.load(f)
    f.close()

    f = open('BernoulliNB_classifier.pickle', 'rb')
    BerNB_classifier = pickle.load(f)
    f.close()

    f = open('LogisticRegression_classifier.pickle', 'rb')
    LR_classifier = pickle.load(f)
    f.close()
    server= Server("http://"+args.couchdb_ip+"/")
#    server = Server('http://115.146.95.99:5984/')
#    db = server['yasmeen-test-tweets']
    db = server[args.dbname]
    max_doc_id =update_couchdb()



