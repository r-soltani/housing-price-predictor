# Main Python program
import pandas as pd
import math


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import datetime
from time import strptime
from matplotlib.dates import DateFormatter
import statsmodels.api as sm


# variables pointing to my data sources
path = 'data/'
inputFile = path + "data-toronto.csv"
inputFile2 = path + "data-predict-toronto.csv"

# features used by all models
allfeatures = ['prime', 'GDP', 'Liberal', 'Conservative', 'CPI', 'spring','summer', 'fall', 'winter'] #''CPI', 'Liberal', 'Conservative', 'spring','summer', 'fall', 'winter']

# Preprocess data
def preprocess(filename):
    print("------------")
    print("Reading file " + filename)
    # read the file
    data = pd.read_csv(filename)

    # create dummy columns based on political fraction column
    dummy = pd.get_dummies(data['political'])
    data = pd.concat([data, dummy], axis=1)

    # create dummy columns based on season column
    dummy2 = pd.get_dummies(data['season'])
    data = pd.concat([data, dummy2], axis=1)


    # optional: create a dummy columns based on something else
    #dummy2 = pd.get_dummies(data['month'])
    #print(dummy2.columns)
    #extra = [str(i) for i in dummy2.columns.values]
    #print(extra)
    #data = pd.concat([data, dummy2], axis=1)

    # artifically add the missing columns to prediction data.
    if (filename == inputFile2):
        data['Liberal'] = 0;
        data['spring'] = 0;
        data['summer'] = 0;
        data['winter'] = 0;


    print(data.head())
    #print(data.head())
    #print(data.info())
    #print(data.describe())
    return data



# analyize data
# create the model using 'data'
# predict new HPI values using 'data2'
def createRegression_SKLearn(data, data2):
    print("Regression using SKLearn...")

    # Features used to create our model and predict new indices
    features = allfeatures

    X = data[features]

    Y = data['index']

    # set an alias
    lm = LinearRegression()


    # divide the data between training data and testing data. test data is 4/10 entire data

    X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.5, shuffle=False)




    # start training using training data
    print("------------")
    print("Training our Model...")
    lm.fit(X_train, y_train)


    # test our model!
    print("------------")
    print("Testing our Model...")
    predictions = lm.predict(X_test)



    # The coefficients
    print("------------")
    print("Comparing the performance of our Model with correct values...")
    print('Coefficients: \n', lm.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, predictions))
    # Explained variance score: 1 is perfect prediction
    print('Variance score (R-Squared): %.2f' % r2_score(y_test, predictions))




    # plot -----------------------
    visualize(y_test, predictions, 'True HPI', 'Predicted HPI', 'Comparing our model test HPI with true HPI')

    # plot -----------------------
    # plot timeseries for the test data from item 84 to item 141

    count = math.floor(len(data[['month']]) * 0.5)
    visualizeTimeSeries(data[['month']][count:], y_test, 'Time', 'HPI', 'True HPI vs Regression Function', predictions)  # plot time vs true HPI, vs regression function


    # show them individually
    #visualizeTimeSeries(data[['month']][85:], y_test, 'Time', 'HPI', 'Time vs Existing HPI') # plot time vs HPI
    #visualizeTimeSeries(data[['month']][85:], predictions, 'Time', 'HPI', 'Time vs Our Model HPI (Regression function)')  # plot time vs HPI

    #visualize(X_test[['prime']], y_test, 'prime', 'HPI', 'Linear Regression Model', X_test[['prime']],  predictions)  # plot prime vs HPI

    # predict  -----------------------
    X_new = data2[features]
    predictions2 = lm.predict(X_new)

    # plot  -----------------------
    #visualize(data2[['prime']], predictions2, 'Prime', 'Predicted HPI', 'Predicting HPI for Nov and Dec 2018') # plot prime vs predicted HPI

    visualizeTimeSeries(data2[['month']], predictions2, 'Time', 'Predicted HPI', 'Predicting HPI for Nov and Dec 2018, using our Model')  # plot  time vs predicted HPI


    output = ""
    return output



# analyize data
# create the model using 'data'
# predict new HPI values using 'data2'
def createRegression_Statsmodel(data, data2):
    print("Regression using Statsmodel...")

    # Features used to create our model and predict new indices
    features = allfeatures

    X = data[features]

    Y = data['index']


    print("------------")
    print("Training our Model...")
    X = sm.add_constant(X)
    lm = sm.OLS(Y, X).fit()



    # test our model!
    print("------------")
    print("Testing our Model...")
    predictions = lm.predict(X)


    # The coefficients
    print("------------")
    print("Comparing the performance of our Model with correct values...")
    print("Summary ", lm.summary())


    # plot -----------------------
    #visualize(y_test, predictions, 'True HPI', 'Predicted HPI', 'Comparing our model test HPI with true HPI')

    # plot -----------------------
    # plot timeseries for the test data from item 84 to item 141
    visualizeTimeSeries(data[['month']], Y, 'Time', 'HPI', 'Time vs HPI', predictions)  # plot time vs true HPI, vs regression function


    # show them individually
    #visualizeTimeSeries(data[['month']][85:], y_test, 'Time', 'HPI', 'Time vs Existing HPI') # plot time vs HPI
    #visualizeTimeSeries(data[['month']][85:], predictions, 'Time', 'HPI', 'Time vs Our Model HPI (Regression function)')  # plot time vs HPI

    #visualize(X_test[['prime']], y_test, 'prime', 'HPI', 'Linear Regression Model', X_test[['prime']],  predictions)  # plot prime vs HPI

    # predict  -----------------------

    X_new = data2[features]

    X_new = sm.add_constant(X_new, has_constant='add')
    predictions2 = lm.predict(X_new)



    # plot  -----------------------
    #visualize(data2[['prime']], predictions2, 'Prime', 'Predicted HPI', 'Predicting HPI for Nov and Dec 2018') # plot prime vs predicted HPI

    visualizeTimeSeries(data2[['month']], predictions2, 'Time', 'Predicted HPI', 'Predicting HPI for Nov and Dec 2018, using our Model')  # plot  time vs predicted HPI
    print(predictions2)

    output = ""
    return output




# visualize data
def visualize(x, y, xlabel, ylabel, title, x2=pd.DataFrame(), y2=pd.DataFrame(), colored='blue'):
    print("------------")
    print("Displaying " + title)
    plt.interactive(False)
    plt.title(title)
    plt.scatter(x, y)
    if not x2.empty:
        plt.plot(x2, y2, color=colored, linewidth=3)
    plt.xlabel(xlabel)  # our truth
    plt.ylabel(ylabel)  # predicted truth
    plt.show()
    return


# visualize timeseries
def visualizeTimeSeries(x, y, xlabel, ylabel, title, y2=[]):
    print("------------")
    print("Displaying " + title)

    fig, ax = plt.subplots()


    #graphdata = pd.Series(y, x)
    #graphdata.plot()



    # read the month column
    # convert the format MMM-YY to datetime.datetime(YYYY, MM, DD, HH, MM)

    x_formatted = [] # result
    for index, item in enumerate(x.values.tolist()):

        # go to inner item
        item = item[0]


        # split and convert to proper datetime format
        items = item.split("-")
        m = strptime(items[1], '%b').tm_mon
        newtime = datetime.datetime(int("20"+items[0]), m, 1, 12, 0)

        # add to new list

        x_formatted.append(newtime)




    import matplotlib.ticker as ticker
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    #ax.set_xticks(x_formatted, minor= "True")
    #ax.set_xticks(np.arange(min(x_formatted), max(x_formatted) + 1, 1.0), minor= "True")
    ax.xaxis.set_major_locator(plt.MaxNLocator(15))

    #ax.set_xticks(np.arange(len(x_formatted)))


    # This simply makes them look pretty by setting them diagonal.
    fig.autofmt_xdate()

    ax.grid(True)

    plt.interactive(False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plot with proper data

    ax.plot_date(x_formatted, y)

    if len(y2) > 0:
        ax.plot(x_formatted, y2)

    plt.show()

    return




def main():
    #preprocess
    data = preprocess(inputFile)
    data2 = preprocess(inputFile2)

    # create regression models
    # use Statsmodel
    output = createRegression_Statsmodel(data, data2)

    # use SKLearn
    output = createRegression_SKLearn(data, data2)

    # we no longer use poly regression
    #output = createPolyRegression(data, data2)


if __name__ == "__main__":
    main()
    print("End of Program.")
