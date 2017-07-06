# Stock Predictor and Portfolio Optimizer

A novice's attempt for (weekly) stock prices prediction and portfolio optimization.

### Technologies used - 
* Python 3
* Scipy
* Scikit-learn
* matplotlib

### Data Source
* [Quandl Data platform](https://www.quandl.com/)

### Major Features
* Prediction of stock prices on weekly basis (predicting next week's prices for a given stock)
* Optimizing portfolio distributions ( optimum ratio of each stock in a portfolio to potentialy maximize profits )

### Algorithms and implementation mechanisms
* Weekly prediction is being done using KNN Regressor using "Bollinger Band Value" and "Simple Moving Average" as input features.
* Stock porfolio optimizer is done for maximizing the "Sharpe Ratio" or "culumative returns", using scipy minimizer (minimizing for -1 * value)

### Testing and results
* This application has been tested for National Stock Exchange, India. The weekly predictions have upto 75 % corelation with the actual results, for the leading (largest market capitalization) stocks.

### TODO
* Add all dependancies in requirements.txt
* Creating web services
* Creating Web-based front-end.
* Improving accuracy by adding more Fundamental and Technical features

### Notes
* As I've mentioned, this implementation is at early stage. If you are an Machine Learning or Stock Market Enthusiast / Expert, feel free to suggest improvements / corrections by creating an issue (or you contact me at vishwajeetvatharkar@gmail.com )
* As I have beginner-level skillset in Python programming language, I might have missed many of the best practices and architectural patterns specific to Python ecosystem, please feel free to suggest some improvements.
* This application was part of my academic project coursework (Major Project, Engineering Final Year, Information Technology)

### Credits
* The algorithmic implementations are inspired by the suggestions given by [Prof. Tucker Bach](http://www.cc.gatech.edu/home/tucker/) in his MOO Course series on Udacity - [Machine Learning for Trading](https://www.udacity.com/course/machine-learning-for-trading--ud501)
