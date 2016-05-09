sentiment analysis required packages


sudo easy_install simplejson 
sudo easy_install jsonpickle 

# nltk 
sudo apt-get install python-nltk

# sklearn
apt-get install build-essential python-dev python-setuptools python-numpy python-scipy libatlas-dev libatlas3gf-base
# or if you have python 3: 
# sudo apt-get install build-essential python3-dev python3-setuptools python3-numpy python3-scipy libatlas-dev libatlas3gf-base

update-alternatives --set libblas.so.3 /usr/lib/atlas-base/atlas/libblas.so.3sudo update-alternatives --set liblapack.so.3 /usr/lib/atlas-base/atlas/liblapack.so.3
apt-get install python-matplotlib

# if it exits then uninstall first using 
# sudo uninstall scikit-learn

sudo pip install -U scikit-learn

# to test sklearn installation run this command
python -c "import sklearn; print sklearn.__version__"
# to test nltk installation, run this commnd
python -c "import nltk; print nltk.__version__"

#Install statmodels for voting classifier using 
easy_install statsmodels


# couchdb with python library
wget http://pypi.python.org/packages/2.6/C/CouchDB/CouchDB-0.8-py2.6.egg

sudo easy_install CouchDB-0.8-py2.6.egg


# nltk used packages
python -c "import nltk; nltk.download('punkt'); nltk.download('treebank');packages =['wordnet','words', 'webtext'];nltk.download(packages);nltk.download('stopwords')"