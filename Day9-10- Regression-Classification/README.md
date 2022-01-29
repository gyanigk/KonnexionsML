# KonnexionsML
###Day 9-10

- There are alot of resources curated for specially practice purposes.
- There are more topics notebooks than u have been taught, so go through

### Topics
- Getting Data
- Git and GitHub
- Pandas
- Numpy, Machine Learning, KNN
- scikit-learn, Model Evaluation Procedures
- Linear Regression
- Logistic Regression, Preview of Other Models
- Model Evaluation Metrics
- Working a Data Problem
- Clustering and Visualization
- Naive Bayes

**Resources:**
* [dataschool blog](https://www.dataschool.io/applying-and-interpreting-linear-regression/)
* [Google Official tutorial notebooks](https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/Index.ipynb)

### Getting Data
* Checking your homework
* Regular expressions, web scraping, APIs ([slides](slides/03_getting_data.pdf), [regex code](code/03_re_example.py), [web scraping and API code](code/03_getting_data.py))
* Any questions about the course project?

**Resources:**
* [regex101](https://regex101.com/#python) is an excellent tool for testing your regular expressions. For learning more regular expressions, Google's Python Class includes an [excellent regex lesson](https://developers.google.com/edu/python/regular-expressions) (which includes a [video](http://www.youtube.com/watch?v=kWyoYtvJpe4)).
* [Mashape](https://www.mashape.com/explore) and [Apigee](https://apigee.com/providers) allow you to explore tons of different APIs. Alternatively, a [Python API wrapper](http://www.pythonforbeginners.com/api/list-of-python-apis) is available for many popular APIs.

### Git and GitHub
* Nick DePrey Guest lecture
* Git and GitHub ([slides](slides/04_git_github.pdf))

**Resources:**
* Read the first two chapters of [Pro Git](http://git-scm.com/book/en/v2) to gain a much deeper understanding of version control and basic Git commands.
* [GitRef](http://gitref.org/) is an excellent reference guide for Git commands.
* [Git quick reference for beginners](http://www.dataschool.io/git-quick-reference-for-beginners/) is a shorter reference guide with commands grouped by workflow.
* The [Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) covers standard Markdown and a bit of "[GitHub Flavored Markdown](https://help.github.com/articles/github-flavored-markdown/)."


### Pandas
* Pandas for data exploration, analysis, and visualization ([code](code/05_pandas.py))
    * [Split-Apply-Combine](http://i.imgur.com/yjNkiwL.png) pattern
    * Simple examples of [joins in Pandas](http://www.gregreda.com/2013/10/26/working-with-pandas-dataframes/#joining)

**Homework:**
* [Pandas homework](homework/05_pandas.md)

**Optional:**
* To learn more Pandas, review this [three-part tutorial](http://www.gregreda.com/2013/10/26/intro-to-pandas-data-structures/), or review these three excellent (but extremely long) notebooks on Pandas: [introduction](http://nbviewer.ipython.org/urls/raw.github.com/fonnesbeck/Bios366/master/notebooks/Section2_5-Introduction-to-Pandas.ipynb), [data wrangling](http://nbviewer.ipython.org/urls/raw.github.com/fonnesbeck/Bios366/master/notebooks/Section2_6-Data-Wrangling-with-Pandas.ipynb), and [plotting](http://nbviewer.ipython.org/urls/raw.github.com/fonnesbeck/Bios366/master/notebooks/Section2_7-Plotting-with-Pandas.ipynb).

**Resources:**
* For more on Pandas plotting, read the [visualization page](http://pandas.pydata.org/pandas-docs/stable/visualization.html) from the official Pandas documentation.
* To learn how to customize your plots further, browse through this [notebook on matplotlib](http://nbviewer.ipython.org/github/fonnesbeck/Bios366/blob/master/notebooks/Section2_4-Matplotlib.ipynb).
* To explore different types of visualizations and when to use them, [Choosing a Good Chart](http://www.extremepresentation.com/uploads/documents/choosing_a_good_chart.pdf) is a handy one-page reference, and Columbia's Data Mining class has an excellent [slide deck](http://www2.research.att.com/~volinsky/DataMining/Columbia2011/Slides/Topic2-EDAViz.ppt).


### Numpy, Machine Learning, KNN
* Numpy ([code](code/06_numpy.py))
* "Human learning" with iris data ([code](code/06_iris_prework.py), [solution](code/06_iris_solution.py))
* Machine Learning and K-Nearest Neighbors ([slides](slides/06_ml_knn.pdf))

**Homework:**
* Read this excellent article, [Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html),(You can ignore sections 4.2 and 4.3.) Here are some questions to think about while you read:
    * In the Party Registration example, what are the features? What is the response? Is this a regression or classification problem?
    * In the interactive visualization, try using different values for K across different sets of training data. What value of K do you think is "best"? How do you define "best"?
    * In the visualization, what do the lighter colors versus the darker colors mean? How is the darkness calculated?
    * How does the choice of K affect model bias? How about variance?
    * As you experiment with K and generate new training data, how can you "see" high versus low variance? How can you "see" high versus low bias?
    * Why should we care about variance at all? Shouldn't we just minimize bias and ignore variance?
    * Does a high value for K cause over-fitting or under-fitting?

**Resources:**
* For a more in-depth look at machine learning, read section 2.1 (14 pages) of Hastie and Tibshirani's excellent book, [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/). (It's a free PDF download!)


### scikit-learn, Model Evaluation Procedures
* Introduction to scikit-learn with iris data ([code](code/07_sklearn_knn.py))
* Exploring the scikit-learn documentation: [user guide](http://scikit-learn.org/stable/modules/neighbors.html), [module reference](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors), [class documentation](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
* Discuss the [article](http://scott.fortmann-roe.com/docs/BiasVariance.html) on the bias-variance tradeoff
* Model evaluation procedures ([slides](slides/07_model_evaluation_procedures.pdf), [code](code/07_model_evaluation_procedures.py))

**Optional:**
* Practice what we learned in class today!
    * If you have gathered your project data already: Try using KNN for classification, and then evaluate your model. Don't worry about using all of your features, just focus on getting the end-to-end process working in scikit-learn. (Even if your project is regression instead of classification, you can easily convert a regression problem into a classification problem by converting numerical ranges into categories.)
    * If you don't yet have your project data: Pick a suitable dataset from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.html), try using KNN for classification, and evaluate your model. The [Glass Identification Data Set](http://archive.ics.uci.edu/ml/datasets/Glass+Identification) is a good one to start with.

**Resources:**
* Here's a great [30-second explanation of overfitting](http://www.quora.com/What-is-an-intuitive-explanation-of-overfitting/answer/Jessica-Su).
* For more on today's topics, these videos from Hastie and Tibshirani are useful: [overfitting and train/test split](https://www.youtube.com/watch?v=_2ij6eaaSl0) (14 minutes), [cross-validation](https://www.youtube.com/watch?v=nZAM5OXrktY) (14 minutes). (Note that they use the terminology "validation set" instead of "test set".)
    * Alternatively, read section 5.1 (12 pages) of [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/), which covers the same content as the videos.
* This video from Caltech's machine learning course presents an [excellent, simple example of the bias-variance tradeoff](http://work.caltech.edu/library/081.html) (15 minutes) that may help you to visualize bias and variance.


### Linear Regression
* Linear regression ([IPython notebook](http://nbviewer.ipython.org/github/justmarkham/DAT4/blob/master/notebooks/08_linear_regression.ipynb))

**Optional:**
* Your optional exercise is to practice what we have been learning in class, either on your project data or on another dataset.

**Resources:**
* To go much more in-depth on linear regression, read Chapter 3 of [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/), from which this lesson was adapted. Alternatively, watch the [related videos](http://www.dataschool.io/15-hours-of-expert-machine-learning-videos/) or read my [quick reference guide](http://www.dataschool.io/applying-and-interpreting-linear-regression/) to the key points in that chapter.
* To learn more about Statsmodels and how to interpret the output, DataRobot has some decent posts on [simple linear regression](http://www.datarobot.com/blog/ordinary-least-squares-in-python/) and [multiple linear regression](http://www.datarobot.com/blog/multiple-regression-using-statsmodels/).
* This [introduction to linear regression](http://people.duke.edu/~rnau/regintro.htm) is much more detailed and mathematically thorough, and includes lots of good advice.
* This is a relatively quick post on the [assumptions of linear regression](http://pareonline.net/getvn.asp?n=2&v=8).


### Logistic Regression, Preview of Other Models
* Logistic regression ([slides](slides/09_logistic_regression.pdf), [exercise](code/09_logistic_regression_exercise.py), [solution](code/09_logistic_regression_class.py))
* Preview of other models

**Resources:**
* For more on logistic regression, watch the [first three videos](https://www.youtube.com/playlist?list=PL5-da3qGB5IC4vaDba5ClatUmFppXLAhE) (30 minutes total) from Chapter 4 of An Introduction to Statistical Learning.
* UCLA's IDRE has a handy table to help you remember the [relationship between probability, odds, and log-odds](http://www.ats.ucla.edu/stat/mult_pkg/faq/general/odds_ratio.htm).
* Better Explained has a very friendly introduction (with lots of examples) to the [intuition behind "e"](http://betterexplained.com/articles/an-intuitive-guide-to-exponential-functions-e/).
* Here are some useful lecture notes on [interpreting logistic regression coefficients](http://www.unm.edu/~schrader/biostat/bio2/Spr06/lec11.pdf).


### Model Evaluation Metrics
* Finishing model evaluation procedures ([slides](slides/07_model_evaluation_procedures.pdf), [code](code/07_model_evaluation_procedures.py))
    * Review of test set approach
    * Cross-validation
* Model evaluation metrics ([slides](slides/10_model_evaluation_metrics.pdf))
    * Regression:
        * Root Mean Squared Error ([code](code/10_rmse.py))
    * Classification:
        * Confusion matrix ([code](code/10_confusion_roc.py))
        * ROC curve ([video](https://www.youtube.com/watch?v=OAl6eAyP-yo))

**Homework:**
* [Model evaluation homework](homework/10_model_evaluation.md), due by midnight on Sunday.
    * [Sample solution code](code/10_glass_id_homework_solution.py).
* Watch Kevin's [Kaggle project presentation video](https://www.youtube.com/watch?v=HGr1yQV3Um0) (16 minutes) for an overview of the end-to-end machine learning process, including some aspects that we have not yet covered in class.
* Read this short article on Google's [Smart Autofill](http://googleresearch.blogspot.com/2014/10/smart-autofill-harnessing-predictive.html), and see if you can figure out exactly how the system works.

**Optional:**
* For more on Kaggle, watch [Kaggle Transforms Data Science Into Competitive Sport](https://www.youtube.com/watch?v=8w4UY66GKcM) (28 minutes).

**Resources:**
* scikit-learn has extensive documentation on [model evaluation](http://scikit-learn.org/stable/modules/model_evaluation.html).
* The Kaggle wiki has a decent page describing other common [model evaluation metrics](https://www.kaggle.com/wiki/Metrics).
* Kevin wrote a [simple guide to confusion matrix terminology](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/) that you can use as a reference guide.
* Kevin's [blog post about the ROC video](http://www.dataschool.io/roc-curves-and-auc-explained/) includes the complete transcript and screenshots, in case you learn better by reading instead of watching.
* Rahul Patwari has two excellent and highly accessible videos on [Sensitivity and Specificity](https://www.youtube.com/watch?v=U4_3fditnWg&list=PL41ckbAGB5S2PavLIXUETzAmi5reIod23) (9 minutes) and [ROC Curves](https://www.youtube.com/watch?v=21Igj5Pr6u4&list=PL41ckbAGB5S2PavLIXUETzAmi5reIod23) (12 minutes).


### Working a Data Problem
* Today we will work on a real world data problem! Our [data](data/ZYX_prices.csv) is stock data over 7 months of a fictional company ZYX including twitter sentiment, volume and stock price. Our goal is to create a predictive model that predicts forward returns.

* Project overview ([slides](slides/11_GA_Stocks.pdf))
    * Be sure to read documentation thoroughly and ask questions! We may not have included all of the information you need...


### Clustering and Visualization
* The [slides](slides/12_clustering.pdf) today will focus on our first look at unsupervised learning, K-Means Clustering!
* The [code](code/) for today focuses on two main examples:
    * We will investigate simple clustering using the iris data set.
    * We will take a look at a harder example, using Pandora songs as data. See [data](data/songs.csv).

**Homework:**
* Read Paul Graham's [A Plan for Spam](http://www.paulgraham.com/spam.html). Here are some questions to think about while you read:
    * Should a spam filter optimize for sensitivity or specificity, in Paul's opinion?
    * Before he tried the "statistical approach" to spam filtering, what was his approach?
    * How exactly does his statistical filtering system work?
    * What did Paul say were some of the benefits of the statistical approach?
    * How good was his prediction of the "spam of the future"?

    * **Confusion matrix:** [Kevin's guide](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/) roughly mirrors the lecture from class 10.
    * **Sensitivity and specificity:** Rahul Patwari has an [excellent video](https://www.youtube.com/watch?v=U4_3fditnWg&list=PL41ckbAGB5S2PavLIXUETzAmi5reIod23) (9 minutes).
    * **Basics of probability:** These [introductory slides](https://docs.google.com/presentation/d/1cM2dVbJgTWMkHoVNmYlB9df6P2H8BrjaqAcZTaLe9dA/edit#slide=id.gfc3caad2_00) (from the [OpenIntro Statistics textbook](https://www.openintro.org/stat/textbook.php)) are quite good and include integrated quizzes. Pay specific attention to these terms: probability, sample space, mutually exclusive, independent.

**Resources:**
* [Introduction to Data Mining](http://www-users.cs.umn.edu/~kumar/dmbook/index.php) has a nice [chapter on cluster analysis](http://www-users.cs.umn.edu/~kumar/dmbook/ch8.pdf).
* The scikit-learn user guide has a nice [section on clustering](http://scikit-learn.org/stable/modules/clustering.html).


### Naive Bayes
* Briefly discuss [A Plan for Spam](http://www.paulgraham.com/spam.html)
* Probability and Bayes' theorem
    * [Slides](slides/13_naive_bayes.pdf) part 1
    * [Visualization of conditional probability](http://setosa.io/conditional/)
    * Applying Bayes' theorem to iris classification ([code](code/13_bayes_iris.py))
* Naive Bayes classification
    * [Slides](slides/13_naive_bayes.pdf) part 2
    * Example with spam email
    * [Airport security example](http://www.quora.com/In-laymans-terms-how-does-Naive-Bayes-work/answer/Konstantin-Tt)
* Naive Bayes classification in scikit-learn ([code](code/13_naive_bayes.py))
    * Data set: [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
    * scikit-learn documentation: [CountVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html), [Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html)
