import numpy as np 
import pandas as pd
import DecisionTree
from DecisionTree import DecisionTree
from sklearn.preprocessing import LabelEncoder, StandardScaler
from future.utils import iteritems
# data can be downloaded from: https://archive.ics.uci.edu/ml/datasets/Mushroom

class RandomForest():
	def __init__(self):
		self.models = []

	def fit(self, X, Y):
		N = len(X)
		d = np.int(len(X[0])*0.5)
		for i in range(N):
			print("Progress:", i, "of ", N)
			sel = np.random.choice(len(X), size = len(X), replace = True)
			Xb, Yb = X[sel], Y[sel]
			model = DecisionTree()
			model.fit(Xb, Yb, d)
			self.models.append(model)

	def predict(self, X):
		N = len(X)
		K = len(self.models)
		output = np.zeros[N,K]
		for i,model in enumerate(self.models):
			output[:,i] = model.predict(X)
		return [np.bincount(output[i]).argmax()for i in range(N)]

NUMERICAL_COLS = ()
CATEGORICAL_COLS = np.arange(22) + 1 # 1..22 inclusive

# transforms data from dataframe to numerical matrix
# one-hot encodes categories and normalizes numerical columns
# we want to use the scales found in training when transforming the test set
# so only call fit() once
# call transform() for any subsequent data
class DataTransformer:
  def fit(self, df):
    self.labelEncoders = {}
    self.scalers = {}
    for col in NUMERICAL_COLS:
      scaler = StandardScaler()
      scaler.fit(df[col].reshape(-1, 1))
      self.scalers[col] = scaler

    for col in CATEGORICAL_COLS:
      encoder = LabelEncoder()
      # in case the train set does not have 'missing' value but test set does
      values = df[col].tolist()
      values.append('missing')
      encoder.fit(values)
      self.labelEncoders[col] = encoder

    # find dimensionality
    self.D = len(NUMERICAL_COLS)
    for col, encoder in iteritems(self.labelEncoders):
      self.D += len(encoder.classes_)
    print("dimensionality:", self.D)

  def transform(self, df):
    N, _ = df.shape
    X = np.zeros((N, self.D))
    i = 0
    for col, scaler in iteritems(self.scalers):
      X[:,i] = scaler.transform(df[col].values.reshape(-1, 1)).flatten()
      i += 1

    for col, encoder in iteritems(self.labelEncoders):
      # print "transforming col:", col
      K = len(encoder.classes_)
      X[np.arange(N), encoder.transform(df[col]) + i] = 1
      i += K
    return X

  def fit_transform(self, df):
    self.fit(df)
    return self.transform(df)

def replace_missing(df):
  # standard method of replacement for numerical columns is median
  for col in NUMERICAL_COLS:
    if np.any(df[col].isnull()):
      med = np.median(df[ col ][ df[col].notnull() ])
      df.loc[ df[col].isnull(), col ] = med

  # set a special value = 'missing'
  for col in CATEGORICAL_COLS:
    if np.any(df[col].isnull()):
      print(col)
      df.loc[ df[col].isnull(), col ] = 'missing'

def get_data():
	df = pd.read_csv('../large_files/mushroom.data', header=None)

	# replace label column: e/p --> 0/1
	# e = edible = 0, p = poisonous = 1
	df[0] = df.apply(lambda row: 0 if row[0] == 'e' else 1, axis=1)

	# check if there is missing data
	replace_missing(df)

	# transform the data
	transformer = DataTransformer()

	X = transformer.fit_transform(df)
	Y = df[0].values
	return X, Y

if __name__ == "__main__":
  X, Y = get_data()
  n = len(X)
  Xtrain, Xtest, Ytrain, Ytest = X[0:n//2], X[n//2:], Y[0:n//2], Y[n//2:]

  # do a quick baseline test
  rf = RandomForest()
  rf.fit(Xtrain, Ytrain)
  Yhat = rf.predict(Xtest)

  print("score:", np.sum(np.array(Yhat)==np.array(Ytest))/len(Ytest))


