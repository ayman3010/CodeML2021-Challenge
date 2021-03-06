# CodeML2021-Challenge 1

This is my solution for challenge 1 for the codeML hackathon by PolyAI, I used a RandomForestRegressor for this solution i got an 92% accuracy but a DecisionTreeRegressor seems to be working just fine with accuracy reaching 88% , I also tried to use a support machine vector but the accuracy was not as good (84 % ), Data normalization didn't seem to do much difference in terms of accuracy.

The file out.csv contains my submission for the challenge.

# Challenge code description
Superconductor critical temperature prevision
This challenge is a regression task for critical temperature prevision.

# Overview
Did you know that some materials can have the resistance that could drop abruptly to zero ? Associating with a magnet, it could levitate. But to obtain this phenomenon, the material should be in a given temperature. This temperature is named the critical temperature (for the superconductor).

The main goal of this challenge is to predict this critical temperature, given some features on these superconductors.

# Data description
The dataset contains about 15 000 superconductors with 80 features. Each category should be explicit enough with its name.

The column to predict is the critical_temp, which gives the critical temperature for the superconductors.

Please note that there could be some missing values, and maybe some outliers.

There are three files available :

train.csv : the training dataset (15 000 rows).
test.csv : the testing dataset (2500 rows).
sample_submission.csv : an example of the submission file.
Note that you should follow the sample_submission.csv to publish your predictions.


# Evaluation
The evaluation metric is the regression (R2) score.




