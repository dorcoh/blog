---
layout: post
title:  "Technical report: Naive methods for estimating the causal effect in multiple treatment setting"
date:   2021-01-03 00:00:00 +0200
categories: causal-inference statistics machine-learning python 
math: true
---

### Introduction 

Causal inference allows us to estimate the causal interactions between random variables. 
It enables us to ask questions such as: "Does some medication lowers blood pressure?". 
Throughout this post I will describe a mini project with the goal of estimating the causal effect in multiple treatment setting 
(e.g., given a number of medications, which one lowers blood pressure most significantly), 
while in the common framework we usually consider binary treatment setting. 

### Background (Potential Outcomes)

Considering the binary treatment setting, Neyman-Rubin's potential outcome framework [2] is defined as follows:


Let \\( X_i \in \\mathbb{R^m} \\) denote the confounders vector (i.e. pre-treatment covariates, or features), 
\\( T_i \in \\{ 0, 1\\} \\) denotes the domain of treatments (e.g., treated and control group), 
and let \\( Y_i \in \\{0,1 \\} \\) denote the outcome, for each unit \\( i= 1,...,N \\)  in our trial.

Following the blood pressure example - $X$ could take patient indicators (e.g., gender, age, etc.),
$T$ indicates whether the patient took medicine, and $Y$ can represent an increase or decrease in blood pressure.

Moreover, let $Y_{i0}$ denote the unit's outcome *had they been subjected to treatment T=0*, and let $Y_{i1}$ denote the unit's 
outcome *had they been subjected to treatment T=1*. These fixed quantities ($Y_0, Y_1$) are known as **potential outcomes** and they 
are never observed together. Therefore, as also noted in [2] - causal inference is a missing data problem, 
and its main challenge is to estimate the missing potential outcome.

To measure the treatment effect for an individual unit $i$ we subtract the outcomes as follows: $Y_{i1}-Y_{i0}$, while 
for estimating the **average treatment effect** we utilize expectation:


$$
ATE \equiv E[Y_1-Y_0]
$$


### Framework for Multiple Treatments


Consider now \\( k \\) treatments that a subject may receive, let \\( \mathbb{T} = \\{ t_1, ..., t_k \\} \\) be the set 
of possible treatments. For each subject, we may observe \\( k \\) different potential outcomes:  $\\{Y_i(t_1), ....Y_i(t_k) \\}$. 
To estimate the causal effect in multiple-treatment setting, the following assumptions are required to hold:
1. **Stable Unit Treatment Value Assumption** (SUTVA) for multiple treatments - i.e. for each subject, 
the set of potential outcomes is identical and there are no "hidden" treatments which may lead to a different unknown outcome.
2. The **consistency assumption** extends to: \\( Y^{obs_i} = I_{i1} * Y_i(t_1) + ... + I_{ik} * Y_i(t_k) \\), 
where $I_{ik}$ denotes an indicator function which gets 1 whether subject $i$ received treatment $k$ and 0 otherwise.

Additionally, we require the treatment assignment mechanism to be individualistic 
(i.e. the $N$ subjects represent a random sample from an infinite super population), and **unconfounded**:

$$
\begin{aligned}
P(T_i = t | X_i, Y_i) \; = \; P(T_i = t | X_i) \equiv \; r(t, X_i)
\end{aligned}
$$

for all units $i=1,...N$. 

#### Generalized Propensity Score

The last term $r(t, X_i)$ is referred to as the *Generalized Propensity Score* (GPS) for 
a subject $i$ and treatment $t \in \mathbb{T} $. Lastly, we demand the assignment mechanism to be probabilistic 
(the **common support** assumption):

$$
\begin{aligned}
\forall t \in \mathbb{T} \; \; P(T_i = t | X_i, Y_i) > 0 
\end{aligned}
$$

When both assumptions hold, we can compare subjects with similar GPS $ R(X) \equiv  (R(t_1, X),...,R(t_k, X))$ 
and expect well estimated effects. Usually, $R(X)$ is unknown, however we may estimate it with any statistical method.

To enforce the common support assumption, one may use the estimate $\hat{R(x)}$ to trim subjects which do not rely in 
the rectangular common support boundaries [1]. Specifically, for each treatment group we measure 
lower and upper bounds for the GPS:

<br/>

$$
\begin{aligned} 
\hat{r}_{min}(t_i, X) = max\{ min(\hat{r}(t_i, X | T_i = t_1)),...,min(\hat{r}(t_i, X | T_i = t_k)) \}

\\\\

\hat{r}_{max}(t_i, X) = min\{ max(\hat{r}(t_i, X | T_i = t_1)),...,max(\hat{r}(t_i, X | T_i = t_k)) \}
\end{aligned}
$$

<br/>

and then we require the following to be satisfied for each $x \in X$:  

$$ \hat{r}(t_i, x) \in (\hat{r}\_{min}(t_i, X),\hat{r}\_{max}(t_i, X) ) $$

The effect for enforcing this assumption can also be illustrated in **Figure 1**. 

<br/>

![Common support]({{ "/assets/multiple-treatment/common_support_multiple.png" | absolute_url }}){: .center-image }
*<b>Figure 1:</b> The common support assumption - we prefer to focus on a sub-population of 
subjects which have an equal chance for receiving all treatments (Scenario **e**). Borrowed from [1]* 

Finally, after dropping non eligible 
subjects, we shall re-estimate $\hat{R(x)}$ to get more accurate propensity scores. We may also repeat this process until 
we are satisfied, however, we should take into account the number of samples left for estimation while repeating this process.


### Dataset

For our experiments, we generated a synthetic classification dataset with 5 treatments, 
where one of them is kept as *control* treatment, and the others are denoted with $t_i$ (where i=1,..,4) for each treatment group. 
In this technical report we chose to estimate the *Average Treatment Effect* (ATE) of $Y(c)$ vs. $Y(t_i)$ where $c$ denotes the treatment of our control group 
and $t_i$ is another treatment, which does not belong to the control group, resulting in a total of 4 ATEs. The ground truth 
ATEs are known, so we could compare truth and predicted values by common metrics, for this task we chose 
*Mean Absolute Error* (MAE). Additionally, we experimented with two versions of datasets, one with all subjects, 
and another with only eligible subjects according to the rectangular common support. 

A histogram of the GPS before and after trimming subjects, can be seen in **Figure 2**. In addition, this figure depicts 
the GPS after re-fitting the model with only eligible subjects. Estimating the GPS was done by multivariate *Logistic Regression*. 

![GPS]({{ "/assets/multiple-treatment/gps_hist.png" | absolute_url }}){: .center-image }
*Propensity scores histograms for (top to bottom): all samples, eligible samples, and eligible samples after refitting their GPS score.*


The full dataset had $46,700$ subjects, with $22$ covariates. After dropping non-eligible units we had $31,984$ units. 
The true ATEs and number of subjects for each treatment group, in each of the datasets, can be observed in **Table 1**. 
Note we have sub-sampled parts of the data, to cause imbalances, so enforcing common support will have some effect. 
To generate the dataset we assisted with [3] framework which provides methods for this goal, 
and [4] for training the classifiers. 

 Treatment Group Key| True ATE (All) |  \# Subjects (All) |  True ATE (Eligible) |  \# Subjects (Eligible)      
|----|----|----|----|----|
control group (c)                  |        0.000000 |        9899 |             0.000000 |        4189 
(control vs.) treatment 1                   |        0.012072 |        9775 |             0.011070 |        5691 
(control vs.) treatment 2                   |        0.025728 |        9484 |             0.028199 |        6064 
(control vs.) treatment 3                   |        0.053787 |        9017 |             0.053498 |        8804 
(control vs.) treatment 4                   |        0.078358 |        8525 |             0.079326 |        7236 

*<b>Table 1:</b> The true ATEs and number of subjects receiving each treatment, across the two versions of our dataset (All, Eligible).* 

### Experiments
To estimate the causal effect, we experimented with two types of methods, the first is known as **Covariate Adjustment**, and the second is **Matching** 
(please note we assume the readers are familiar with these methods, otherwise - there are plenty of sources to learn more about them). 
Our intention was to check whether naive methods (i.e. which were intended to handle binary treatment settings) 
were sufficient for estimating the causal effect in multiple treatment setting. For evaluating the covariate 
adjustment methods, we have used once all data, and then only eligible samples (mentioned in **Table 2** as *All*, and *Common support*). 
For the evaluation of matching algorithms, we have used only the eligible samples, mainly for computational reasons. 

#### Covariate Adjustment results

Dataset               | Features            | Learner          |  Classifier             |       MAE
|---|---|---|---|---|
All samples | Covariates + Propensity scores | s\_learner | LogisticRegression |  0.040844 
               |            |           | SVC |  0.041728 
               |            |           | RandomForestClassifier |  0.042789 
               |            |           | MLPClassifier |  0.039683 
               |            | t\_learner | LogisticRegression |  0.021183 
               |            |           | SVC |  0.011573 
               |            |           | RandomForestClassifier |  0.010071 
               |            |           | MLPClassifier |  0.018576 
               | Propensity scores | s\_learner | LogisticRegression |  0.040844 
               |            |           | SVC |  0.041728 
               |            |           | RandomForestClassifier |  0.042688 
               |            |           | MLPClassifier |  0.039253 
               |            | t\_learner | LogisticRegression |  0.021183 
               |            |           | SVC |  0.011573 
               |            |           | RandomForestClassifier |  0.012261 
               |            |           | MLPClassifier |  0.018506 
               | Covariates | s\_learner | LogisticRegression |  0.040844 
               |            |           | SVC |  0.041728 
               |            |           | RandomForestClassifier |  0.042915 
               |            |           | MLPClassifier |  0.040693 
               |            | t\_learner | LogisticRegression |  0.021183 
               |            |           | SVC |  0.011573 
               |            |           | RandomForestClassifier |  0.010837 
               |            |           | MLPClassifier |  0.019593 
Common support | Covariates + Propensity scores | s\_learner | LogisticRegression |  0.041382 
               |            |           | SVC |  0.042266 
               |            |           | RandomForestClassifier |  0.043377 
               |            |           | MLPClassifier |  0.040725 
               |            | t\_learner | LogisticRegression |  0.020985 
               |            |           | SVC |  0.012611 
               |            |           | RandomForestClassifier |  0.010289 
               |            |           | MLPClassifier |  0.017955 
               | Propensity scores | s\_learner | LogisticRegression |  0.041382 
               |            |           | SVC |  0.042266 
               |            |           | RandomForestClassifier |  0.043427 
               |            |           | MLPClassifier |  0.042266 
               |            | t\_learner | LogisticRegression |  0.020985 
               |            |           | SVC |  0.012611 
               |            |           | RandomForestClassifier |  **0.009256** 
               |            |           | MLPClassifier |  0.017843 
               | Covariates | s\_learner | LogisticRegression |  0.041382 
               |            |           | SVC |  0.042266 
               |            |           | RandomForestClassifier |  0.043074 
               |            |           | MLPClassifier |  0.041458 
               |            | t\_learner | LogisticRegression |  0.020985 
               |            |           | SVC |  0.012611 
               |            |           | RandomForestClassifier |  0.010359 
               |            |           | MLPClassifier |  0.018881 

*<b>Table 2:</b> Covariate adjustment results, Random Forest classifier with propensity scores as features beats other methods, 
when evaluated on the common-support enforced dataset.*

#### Matching results


Features                               | Distance type                   | Calipher               |       MAE 
|---|---|---|---|
Propensity scores | cityblock\_distance |  None |  0.092390
                               |                    |  0.05 |  0.046972 
                               |                    |  0.1 |  0.064699 
                               | euclidean\_distance |  None |  0.084655 
                               |                    |  0.05 |  0.064810 
                               |                    |  0.1 |  0.079215 
Covariates | cityblock\_distance |  None |  0.007786 
                               | euclidean\_distance |  None |  **0.005616** 
Covariates + Propensity scores | cityblock\_distance |  None |  0.007876 
                               | euclidean\_distance |  None |  0.005878                          
Covariates | cityblock\_distance |  10 | **0.005496** 
                               | euclidean\_distance |  2.5 |  0.010433 
                               |                    |  5 |  0.005524 
                               |                    |  10 |  0.005616 
Covariates + Propensity scores | cityblock\_distance |  10 |  0.005878 
                               | euclidean\_distance |  2.5 |  0.010779 
                               |                    |  5 |  0.005788 
                               |                    |  10 |  0.005878 

*<b>Table 3:</b> Matching results - nearest neighbor algorithms based on covariates as features get the best results.*

### Conclusions | Future work
Through the results, we have seen that generally (or at least for the eligible samples dataset) matching algorithms 
outperform covariate adjustment. Perhaps the reason is that covariate adjustment method requires the model to be well specified. 
In **Table 2** we observe that T-Learners perform better than S-Learners [5]. Additionally, in **Table 3** we have observed
that matching on covariates or, propensity scores plus covariates is better than matching based only on propensity scores. 

This technical report provides some empirical evidence that methods which were intended to handle binary treatment work well also in multiple treatment setting. 
To further check our conclusions, we may experiment with additional types of datasets, possibly with larger treatments domain. 
An interesting question is whether the effectiveness of these methods remain decent even when increasing the number of treatments. 
As follows from the results and in the spirit of [6], extending the experiments with matching algorithms may be a good future direction as well. 
Specifically, we can match subjects with other distances types (e.g., Mahalanobis distance), and choose other methods for enforcing common support. 
Then we could compare these results to the specialized (multi treatment) matching algorithms introduced in [6]. 
Another method which we have not explored yet, is to train $k$ causal forests [7] for each treatment group, which can be 
compared to [8] who provided an extension for causal forest to multiple treatment setting.

### Acknowledgment

This work was completed as a final project within "Introduction to Causal Inference" course by Uri Shalit and Rom Gutman, which took place at the Technion. 

### References
1. Lopez, Michael J., and Roee Gutman. "Estimation of causal effects with multiple treatments: a review and new ideas." Statistical Science 32.3 (2017): 432-454.

2. Sekhon, Jasjeet S. "The Neyman-Rubin model of causal inference and estimation via matching methods." The Oxford handbook of political methodology 2 (2008): 1-32.

3. Chen, H., Harinen, T., Lee, J.-Y., Yung, M., and Zhao, Z. (2020). Causalml:
Python package for causal machine learning.

4. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel,
O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas,
J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay,
E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine
Learning Research, 12:2825–2830.

5. Künzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). Metalearners for estimating heterogeneous treatment 
effects using machine learning. Proceedings of the national academy of sciences, 116(10):4156–4165.

6. Scotina, A. D. and Gutman, R. (2019). Matching algorithms for causal
inference with multiple treatments. Statistics in medicine, 38(17):3139–
3167.

7. Wager, S. and Athey, S. (2018). Estimation and inference of heterogeneous
treatment effects using random forests. Journal of the American Statistical
Association, 113(523):1228–1242.

8. Lechner, M. (2018). Modified causal forests for estimating heterogeneous
causal effects. arXiv preprint arXiv:1812.09487.
