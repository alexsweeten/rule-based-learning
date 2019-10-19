# Rule Based Learning for Transcriptional Regulation
<img src="https://raw.githubusercontent.com/hackseq/hackseq_graphics/master/0_logo/hackseq_logo.png">
This is the GitHub repo for the hackseq19 project: Rule Based Learning for Transcriptional Regulation! 

## Rationale
Gene regulatory sites, such as Transcription Factor Binding Sites (TFBS's) and Promoters, are extremely important regions within both eukaryotic and prokaryotic genomes. Predicting whether or not a site acts as a regulatory element is an important, yet surprisingly difficult task. There has been a lot of focus in recent years towards building machine learning (ML) approaches for automatically detecting these genomic regions. In this hackathon, we hope to experiment with some of these tools. 

## Goals
Our goals during hackseq19 are to:

* a) Build an accurate classifier for a given gene regulation dataset.
* b) Build an interpretable classifier that outputs useful rules, describing each dataset.

We will experiment with many different classifiers, including decision trees, random forests, support vector machines, and neural networks. Accuracy is measured using F1 score, which we can visualize on our leaderboard (see below). Interpretability is measured by how clearly we can deduce rules from our dataset. An example rule:

`IF  Position[2] == "G" AND Position[3] == "C" THEN Class == "TFBS"`

## Data

Our [leaderboard page is available here](http://spheric-alcove-256103.appspot.com/). You are required to sign in using your Google account. Once signed in, you can choose your username and submit files to the leaderboard. The leaderboard is based on a hacked version of my [Natural Language Processing course professor's website](http://anoopsarkar.github.io/nlp-class/). 

<b>Datasets:</b><br />
* 1 Human Chromosome #1 TFBS
* 1 Ecoli K12 TFBS
* 2 Ecoli K12 Promoter Region
* 1 Pokemon

These come from a variety of sources, including gene regulation databases and previous Kaggle competitions.<br />

## Results

The following graph represents our progress improving classifier accuracy over the course of hackseq19. x-axis is measure in hours of time, y-axis is measure in terms of F1 Score. Dashed lines represent our "oracle", representing the highest recorded accuracy in the literature. As you can see, we beat the oracle score for Huamn SP1 TFBS! 

<img src="https://github.com/SweetiePi/rule-based-learning/blob/master/plots/plot.png" width="800" height="550">

## Team Members

<b>Team Lead:</b><br />
Alex Sweeten <br /><img src="https://raw.githubusercontent.com/SweetiePi/asweeten.github.io/master/alex.gif" width="180" height="180" />

<b>Participants:</b><br />
Aris Grout <br /><img src="https://raw.githubusercontent.com/SweetiePi/asweeten.github.io/master/birthday.gif" width="275" height="275" />

Chahat Upreti <br /><img src="https://raw.githubusercontent.com/SweetiePi/asweeten.github.io/master/coffee.gif" width="275" height="275" />

Jade Chen <br /><img src="https://raw.githubusercontent.com/SweetiePi/asweeten.github.io/master/corrosive.gif" width="275" height="275" />

Kate Gibson <br /><img src="https://raw.githubusercontent.com/SweetiePi/asweeten.github.io/master/headphones.gif" width="275" height="275" />

Oriol Fornes <br /><img src="https://raw.githubusercontent.com/SweetiePi/asweeten.github.io/master/hot.gif" width="275" height="275" />

Priyanka Mishra <br /><img src="https://raw.githubusercontent.com/SweetiePi/asweeten.github.io/master/lemon.gif" width="275" height="275" />

Shawn Hsueh <br /><img src="https://raw.githubusercontent.com/SweetiePi/asweeten.github.io/master/love.gif" width="275" height="275" />

Zakhar Krekhno <br /><img src="https://raw.githubusercontent.com/SweetiePi/asweeten.github.io/master/spin.gif" width="275" height="275" />
