<h1> Unsupervised UFOs </h1>

<h6>Team: Emily Quigley, Paul Sandoval, Joe Shull, Scott Wigle </h6>

<h6> Data Set: National UFO Reporting Center </h6>

<h3> Team Breakdown </h3>
Paul & Joe (Data Cleaning & EDA team)
<br>
Scott & Emily (Modeling Team)

<h3> Data Cleaning </h3>
![](images/ufo_class.png)
<br>
<br>
![](images/parse_html.png)
<br>
<br>

<h3> Exploratory Data Analysis </h3>
![](images/occurences_by_month.png)
<br>
<br>
![](images/Figure_1.png)



<h3> Data Structure & Sharing Data</h3>
Data cleaning team scrubbed the data and shared in CSV format using Git.
Data modeling team wrote one NLP class which did the initial work on our data to get it ready for modeling. Secondly, we wrote a modeling class that would do our modeling and plotting.
<br>
![](msno.png)


<h3> Modeling </h3>
1. CountVectorizer to remove punctuation, lowercase, etc.
<br>
<br>
2. Tf-idf vectorization
<br>
<br>
3. Started with TruncatedSVD from SK Learn ran into some complications.
We used explained_variance_ratio_ looked like we may have scaled incorrectly
<br>
<br>
4. Scree plot shows variance described by principal components. We saw the plot below and knew something wasn't right.
<br>
<br>
![](images/First_sreePlot_with_10_topics_no_lemm.png)
<br>
<br>
5. We were trying to plot a scree plot and another plot to show our clusters similar to PCA from the class assignment. Since SVD is related to PCA we thought we could do this with SVD too. When things weren't working we switched to PCA to try and fix to our issue due to lack of time.
<br>
<br>
![](images/final_scree.png)
<br>
<br>
6. Below are our top features from our top 10 topics
<br>
<br>
![](images/10_topicwords.png)

<h3> Joe's Attempt At Greatness </h3>
Yes, this involves connecting to an API.

![](images/heatmap_attempt.png)

### Complications
* Lemmatizing and using CountVectorizer
* Method for handling large data to speed up processing
