#import packages
import pandas as pd
import numpy as np

# Pandas profiling allows you to understad your dataframe with basic stats on a summary report
from pandas_profiling import ProfileReport

# Lable Encoder allows you to convert a string column to a binary / numerical values
from sklearn.preprocessing import LabelEncoder

# standard Scaler allows you in take few columns with values and transferr their values into a scalable values for machine learning models
from sklearn.preprocessing import StandardScaler

# train test split allows you to split your dataframe into two data frames. one for building the model (train) and one to test your model (test)
from sklearn.model_selection import train_test_split

# importing Linear Rergression model
from sklearn.linear_model import LinearRegression



df = pd.read_csv('StudentsPerformance.csv')

#check data frame
df.head(5)


#describe dataframe
df.describe()



##first we need to convert all the string variables into numerical variables - we can do that with the labelEncoder
#create an encoder
encoder = LabelEncoder()

#convert gender string values into numeric values
df['gender'] = encoder.fit_transform(df['gender'])
#Store the labeling string values for later use if you neet to undarstand who is female and who is male
gender_mappings = {index: label for index, label in enumerate(encoder.classes_)}

#Do the same processes to all the measure values you want to convert into numerical variables
df['race/ethnicity'] = encoder.fit_transform(df['race/ethnicity'])
race_mappings = {index: label for index, label in enumerate(encoder.classes_)}

df['parental level of education'] = encoder.fit_transform(df['parental level of education'])
parental_edu_mappings = {index: label for index, label in enumerate(encoder.classes_)}

df['lunch'] = encoder.fit_transform(df['lunch'])
lunch_mappings = {index: label for index, label in enumerate(encoder.classes_)}

df['test preparation course'] = encoder.fit_transform(df['test preparation course'])
preparation_mappings = {index: label for index, label in enumerate(encoder.classes_)}

#check one of your mapping
print(gender_mappings)
print(race_mappings)
print(parental_edu_mappings)
print(lunch_mappings)
print(preparation_mappings)

#chack the new look of the dataframe
df.head(5)



#creating a profile report of our data
#this will show data on every column, and show correlations, missing values and more.
report = ProfileReport(df)

#show the report
report

#in the report go to correlation. we see that in the correlation,
#the predictive values are three columns only: math score, reading score and writhing score.
#we can create a new dataframe with just these columns and then run the Linear Regression model



# so now we want to predict math score based on reading / writing score
# in addition, we want to predict reading score based on math / writing score
# lastly, we want to predict writing score based on math / reading score

## let's select just the wanted columns
X = df[['math score','reading score','writing score']]
X



## before we split our data - we need to scale every column in order to make each column proportional to the other columns.
# after scaling each column will have mean = 0 and st = 1.

#first we create our scaler
scaler = StandardScaler()

#second we need to change our dataframe values into a scalable values
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

#check the new scalable dataframe
X



## further we are going to split our x into three different models, and we also going to get three different y's.

#first let's create our three different y - THESE VALUES ARE GOING TO BE OUR VALUES WE WANT TO PREDICT
y_math = X['math score']
y_reading = X['reading score']
y_writing = X['writing score']

#second let's create our three different x - THESE VALUES ARE GOING TO BE OUR PREDICTIVE VALUES
x_math = X[['reading score','writing score']]
x_reading = X[['math score','writing score']]
x_writing = X[['reading score','math score']]




##now let's split our data into train and test data sets
x_math_train, x_math_test, y_math_train, y_math_test = train_test_split(x_math, y_math, train_size = 0.7)
x_reading_train, x_reading_test, y_reading_train, y_reading_test = train_test_split(x_reading, y_reading, train_size = 0.7)
x_writing_train, x_writing_test, y_writing_train, y_writing_test = train_test_split(x_writing, y_writing, train_size = 0.7)

#check one of our dataset.
#the whole dataframe = 1000. since our train size = 0.7, the data train set will be 700 rows = 70%.
x_math_train




##now we want to use Linear Regression model in order to predict all out y test datasets
#creating three models for each train and test data sets
math_model = LinearRegression()
reading_model = LinearRegression()
writing_model = LinearRegression()

#fit the models to the respective datasets
math_model.fit(x_math_train, y_math_train)
reading_model.fit(x_reading_train, y_reading_train)
writing_model.fit(x_writing_train, y_writing_train)

#see the tested linear regression scores
math_r2 = math_model.score(x_math_test, y_math_test)
reading_r2 = math_model.score(x_reading_test, y_reading_test)
writing_r2 = math_model.score(x_writing_test, y_writing_test)

#check the scores
print(f"math_r2: {math_r2}")
print(f"reading_r2: {reading_r2}")
print(f"writing_r2 : {writing_r2}")

