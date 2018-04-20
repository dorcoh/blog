---
layout: post
title:  "Research project: CNF formula entropy approximation patch for STS explained"
date:   2017-06-19 00:00:00 +0200
categories: sat c++ 
math: true
---
I describe here a patch I've made to SearchTreeSampler - an approximate SAT model counter by [Stefano Ermon](https://cs.stanford.edu/~ermon/).

The [patch](https://github.com/dorcoh/hashing-optimization) leverages the uniform solution sampling method to compute the entropy of CNF formula, a new property of SAT formulas which I'll define below.

# Preliminaries

## SAT

Let \\( X_1 ,... X_n \\) be boolean variables

A boolean formula \\( \varphi \\) is said to be in CNF if it's a logical conjunction of a set of clauses \\( C_1,...,C_n \\), where each clause \\( C \\) is a logical disjunction of a set of literals (a base element that could be a variable or it's negation). A clause for example: \\( (x_1 \vee \neg x_2) \\)

SAT problem is defined as deciding whether there exists an assignment that satisfies \\( \varphi \\)

An example: \\( \varphi=(x_1 \vee \neg x_2 \vee x_3)\wedge(x_4 \vee \neg x_1)\wedge(x_2 \vee \neg x_3) \\)

Satisfying assignment: \\( \{ x_1=1,x_2=1,x_3=1,x_4=1 \} \\)

\\( \Longrightarrow \varphi \\) is SATISFIABLE

## Model counting

Let \\( V \\) be the set of boolean variables of \\( \varphi \\) and let $$ \sum $$ be the set of all possible assignments to these variables

An assignment \\( \sigma \in \sum \\) is a mapping that assigns a value in \\( \{0,1\} \\) to each variable in \\( V \\)

Define the weight \\( w(\sigma) \\) to be 1 if \\( \varphi \\) is satisfied by \\( \sigma \\) and 0 otherwise

In this context $$  W=\sum_{\sigma \in \sum}^{} w(\sigma)=\#(\varphi)  $$ is the actual number of solutions to  \\( \varphi \\)

## Entropy

Let $$ \varphi $$ be a propositional CNF formula, $$ var(\varphi) $$ its set of variables and $$ lit(\varphi) $$ its set of literals. 

If $$ \varphi $$ is SATISFIABLE, we denote by $$ r(l) $$, for $$ l \in lit(\varphi) $$, the ratio of solutions to $$ \varphi $$ that satisfy $$ l $$. Hence for all $$ v \in var(\varphi) $$ , it holds that $$ r(v) + r(\bar v) = 1 $$

The entropy of a variable $$ v \in V $$ is defined by:

$$ e(v) = -r(v)logr(v) -r(\bar v)logr(\bar v) $$ 

_The entropy of a satisfiable formula is the average entropy of its variables_

Entropy function is depicted below:
![entropyShan]({{site.baseurl}}/_posts/entropyShan.png)

![entropyShan.jpg]({{site.baseurl}}/_posts/entropyShan.jpg)

**Motivation**: Since model counting is a $$ #P $$ problem, entropy is hard to compute and requires $$ n $$ calls to a model counter (as the number of variables in the input formula).

# STS and patch explained

## Hashing and optimization

STS, Search tree sampler is an approximate model counter designed by Stefano Ermon. It uses hashing and optimization technique in order to count solutions. Briefly, in the context of model counting, hashing means that on each 'level' the algorithm explores, we shrink the solutions space. Optimization means using a SAT  solver as an oracle to tell the algorithm if solutions still exist after shrinking. This technique is also used in probablistic inference problems.

Specifically STS works by sampling uniform (controlled by a parameter) solutions. I took advantage of this mechanism and on each run of the algorithm I recorded those uniform solutions, in order to cheaply approximate the entropy, with only one run of STS instead of $$ n $$ runs - as the size of the formula. Computing the entropy requires to compute $$ r(v) $$ for each literal, $$ r(v) $$ is the ratio of solutions that the literal $$ v $$ appears in, out of all formula's solutions. So technically if we have a decent amount of uniform solutions, we can approximate the variables entropy.

## Patch explained

STS is built on top of Minisat, the algorithm is implemented on `Main.cc`

In the following I'll describe my patch:

First I added the needed variables

```cpp
std::vector<int> varSolCountPos;	// counter for pos
std::vector<int> varSolCountNeg;	// counter for neg
std::vector<double> rvPos;		// vector for r(v)
std::vector<double> rvNeg;
std::vector<double> ev;			// vector for e(v)
std::vector<double> entropy;		// vector for formula entropy for each sample
```

Then I initialized them:

```cpp
varSolCountNeg.resize(var_num);
varSolCountPos.resize(var_num);
rvPos.resize(var_num);
rvNeg.resize(var_num);
ev.resize(var_num);
for (int iter=0; iter<var_num; iter++)
{
  varSolCountPos[iter] = 0;
  varSolCountNeg[iter] = 0;
  rvPos[iter] = 0;
  rvNeg[iter] = 0;
  ev[iter] = 0;
}
```

I used the loop for outputting solutions to count the number of times each literals appear in the solutions, so I added the following lines:

```cpp
// compute #(x) and #(!x)
if (OutputSamples[l][i] == 1)
{
	varSolCountPos[i] += 1;
} else {
	varSolCountNeg[i] += 1;
}					
```

Added an option for printing the counts nicely (controlled by verb parameter)

```cpp
// print literals solution counters
if (verb>1)
{
// print pos - #(!x)
for (int iter=0; iter < var_num; iter++)
{
  if (iter!=var_num-1)
  {
  	if (verb>0)						
  		printf("%d,",varSolCountPos[iter]);							
  }
  else
  {
  	if (verb>0)						
  		printf("%d\n",varSolCountPos[iter]);							
}

}
// print neg - #(x)
for (int iter=0; iter < var_num; iter++)
{
  if (iter!=var_num-1)
  {
    if (verb>0)						
    	printf("%d,",varSolCountNeg[iter]);							
  }
  else
  {
    if (verb>0)						
    	printf("%d\n",varSolCountNeg[iter]);							
  }
 }
}
```

Now when I have the counts I can compute $$ r(v) $$ and the entropy of each variable:

```cpp
for (int iter=0; iter < var_num; iter++)
{
  int total = varSolCountPos[iter] + varSolCountNeg[iter];
  double logrv = 0;
  double logrvBar = 0;
  rvPos[iter] = (double)varSolCountPos[iter] / total;
  rvNeg[iter] = 1-rvPos[iter];
  if (rvPos[iter] != 0 && rvNeg[iter] !=0)
  {
    logrv = log2(rvPos[iter]);
    logrvBar = log2(rvNeg[iter]);
  } 
  else 
  {
    if (rvPos[iter] == 0)
      logrv = 0;
    if (rvNeg[iter] == 0)
      logrvBar = 0;
  } 
  ev[iter] = -( (rvPos[iter]) * (logrv) ) - ( (rvNeg[iter])*(logrvBar) );
}
```

Computing the formula entropy is done by averaging the variables entropy:

```cpp

double sumEntropy = 0;
for (int iter=0; iter < var_num; iter++)
{
	sumEntropy += ev[iter];
}

entropy[ss] = sumEntropy / var_num;
printf("entropy=%lf", entropy[ss]);
 ```
 
 And finally printing the averaged entropy outside the loop (the algorithm runs `nsample` times)
 
 ```cpp
 double avgEntropy = 0;
 for (int iter=0; iter<nsamples; iter++)
 {
 	avgEntropy += entropy[iter];
 }
 avgEntropy = (double)avgEntropy / nsamples;
 printf("Average Entropy: %f\n", avgEntropy);
 ```

## Evaluation

I describe one easy formula (100 variables 400 clauses) results, which has an exact entropy (computed by exact model counter Cachet) of 0.673665 .

With the default parameters (k=50, nsamples=10) of STS I got: 0.610541
The parameter k controlles the uniformity of the samples, the higher it is we have stronger guarantee that the solutions are uniform. nsamples is the number of times the algorithm runs.

Increasing k didn't improve the result (at least for easy formulas like this one)

Increasing nsamples on the other hand, got me near the actual entropy:

nsamples=20 , entropy=0.647872

nsamples=40 , entropy=0.664543

nsamples=80 , entropy=0.673098

## References
-Stefano Ermon, Carla Gomes, and Bart Selman.
Uniform Solution Sampling Using a Constraint Solver As an Oracle.
UAI-12. In Proc. 28th Conference on Uncertainty in Artificial Intelligence, August 2012.
