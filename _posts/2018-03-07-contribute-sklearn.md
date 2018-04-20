---
layout: post
title:  "Tutorial: Contribute to sklearn, or other open-source projects"
date:   2018-03-07 00:00:00 +0200
categories: python machine-learning github opensource 
---

Lately I've been intersted in understanding more deeply some of the modules in `sklearn` library. Hence I decided the best way to learn is to contribute, which also helps the community. In this post I summarized the steps to be taken to work on a new pull request. I assumed you've already done initial setup, if not - take a look at <b>Get Started</b> section. The post goal is to make the contribution process easier for newcomers, we will examine and fix together a real bug.

# Get started

Fortunately, there are plenty of tutorials to get started in contributing to open source projects. First reference is to `sklearn` official [contributing page][sklearn-contribute]. Each open source project has its own development guidelines, familiarize yourself with them.

Two more tutorials are suggested there, the first is from `NumPy` documentation about setting your [development workflow][numpy-devel]. Second is from `astropy` docs, very similar [tutorial][astropy-devel]. I highly recommend following one of them (or both), in this blog post I assume you've set your development enviroment, preferably using `NumPy`'s article.

To save you time, I summarize the steps to get started:
* [Set up your fork][setup-fork]
* [Create a new feature branch][feature-branch] (also described below)
* Start hacking

# Virtual Enviroment

For better workflow I use virtual enviroments, basically they help us manage our dependencies and avoid conflits. There are few choices available, I use `conda`. To list enviroments available on your machine (maybe you've forgotten them) type:

` conda info -e `

it should show a list as the following:

```
base                  *  /home/deebee/anaconda2
scikit                   /home/deebee/anaconda2/envs/scikit

```

To set the enviroment type `source activate enviroment_name`. Each time you use `pip install` or `conda install` the packages would be installed onto this enviroment only. When you finished working, to exit the enviroment type `source deactivate`.

<b>Note:</b> First time you'll have to define a new enviroment, check out the tutorials mentioned. For Python 3.6 this can be done by `conda create -n enviroment_name python=3.6 anaconda`.

# Retrieve code & create branch

Each time you start working on code, you should fetch the latest repository (to avoid conflicts when you'd like merging to master). You can perform this by:
`git fetch upstream`

To create a new branch based on `master` type:
`git checkout -b our-new-feature upstream/master`

or switch into an existing one: 
`git checkout our-new-feature`

Next step is installing the branch, you can type:
`pip install --editable .`

then whenever source code of compiled extension changes type:
`python setup.py build_ext --inplace`

<b>Note:</b> This steps assumes you've created `upstream`,
For more about it take a look [here][retrieve-code].

## Make changes

# Explore and understand the bug

From here on I'll describe a conrecte example, an actual bug that I'd like to fix.

Specifically, there's a [bug][issue] with `CountVectorizer`. This class has a property `ngram_range` which allows to choose range for extracting n-grams (e.g., 1 <= n <= 2 extracts 1-gram and 2-grams), the problem is a missing validation for this parameter, it turns out that we could set `ngram_range=(2,1)`, which doesn't make sense of course. Following `sklearn` current guidelines, we should perform the validation on `fit(..)` function rather than `__init__(..)` - the class constructor. 

Before we dive into fixing this, I'll give you a usage example for `CountVectorizer`, for those of you who aren't familiar with this module.

<h3>CountVectorizer usage example here</h3>

Let's explore the corresponding module `feature_extraction/text.py`, take a look at these class definitions:

```python
class HashingVectorizer(BaseEstimator, VectorizerMixin, TransformerMixin)
class CountVectorizer(BaseEstimator, VectorizerMixin)
class TfidfVectorizer(CountVectorizer)
```

All of them implement `fit(..)` and has the `ngram_range` property, however `TfidfVectorizer`, as well as its `fit(..)`  is inherited from `CountVectorizer`, making our life a bit easier.

# Add test

The next step is deciding on the expected behaviour when encountering this bug. We can reproduce the bug using a custom script, or Python console.

However, the more standard way is to write a unit test (`sklearn` uses `PyTest` for this purpose), which should fail at this moment. Test scripts are usually stored in `our_module/tests` directory, , let's run tests for `feature_extraction` module:

```python
(scikit) deebee@realm:~/Desktop/open-source/scikit-learn/sklearn/feature_extraction$ pytest tests/test_text.py 
============================================ test session starts ============================================
platform linux2 -- Python 2.7.14, pytest-3.4.0, py-1.5.2, pluggy-0.6.0
rootdir: /home/deebee/Desktop/open-source/scikit-learn, inifile: setup.cfg
collected 45 items                                                                                          

tests/test_text.py .............................................                                      [100%]

=================================== 45 passed, 2 warnings in 2.65 seconds ===================================
```

It passed, now let's add our new test function to `test_text.py`:

```python
@pytest.mark.parametrize("vec", [
        HashingVectorizer(ngram_range=(2, 1)),
        CountVectorizer(ngram_range=(2, 1)),
        TfidfVectorizer(ngram_range=(2, 1))
    ])
def test_vectorizers_invalid_ngram_range(vec):
    # vectorizers could be initialized with invalid ngram range
    # test for raising error message
    invalid_range = vec.ngram_range
    message = ("Invalid value for ngram_range=%s "
               "lower boundary larger than the upper boundary."
               % str(invalid_range))

    assert_raise_message(
        ValueError, message, vec.fit, ["good news everyone"])
    assert_raise_message(
        ValueError, message, vec.fit_transform, ["good news everyone"])

    if isinstance(vec, HashingVectorizer):
        assert_raise_message(
            ValueError, message, vec.transform, ["good news everyone"])
```
 
<b>Explanation:</b> First, we use `@pytest.mark.parametrize` to make this a parametric test (or else we'll have to define an object for each of the tested classes). Then I initialize the objects with faulty `ngram_range`, and check if it raises the corresponding error and message while calling `fit` or `fit_transform` (notice only `HashingVectorizer` implements `transform` function).

When running the test again, it should fail (I skipped some of the output for readability):

```python
(scikit) deebee@realm:~/Desktop/open-source/scikit-learn/sklearn/feature_extraction$ pytest tests/test_text.py 
.
.
.
../utils/testing.py:402: AssertionError
============================== 1 failed, 45 passed, 2 warnings in 2.61 seconds ==============================
```

You can read more about unit tests in Python docs, or this wonderful blog (Hitchhikers guide..TODO: ADD LINK)

Now we are ready to apply the fix.

# Fix

Applying the fix was a two-step process, first I defined a helper function for validation under `VectorizerMixin` class (which all relevant classes inherit from):

```python

def _validate_params(self):
    """Check validity of ngram_range parameter"""
    min_n, max_m = self.ngram_range
    if min_n > max_m:
        raise ValueError(
            "Invalid value for ngram_range=%s "
            "lower boundary larger than the upper boundary."
            % str(self.ngram_range))


```

Then I called it in relevant functions of each class, I describe here only one of the fixed functions (you can chek other calls in `feature_extraction.text` module) which is `HashingVectorizer.fit(..)`:

```python
def fit(self, X, y=None):
    """Does nothing: this transformer is stateless."""
    # triggers a parameter validation
    if isinstance(X, six.string_types):
        raise ValueError(
            "Iterable over raw text documents expected, "
            "string object received.")

    self._validate_params()

    self._get_hasher().fit(X, y=y)
    return self
```

[sklearn-contribute]: http://scikit-learn.org/stable/developers/contributing.html
[numpy-devel]: https://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html
[astropy-devel]: http://astropy.readthedocs.io/en/latest/development/workflow/development_workflow.html
[retrieve-code]: http://scikit-learn.org/stable/developers/contributing.html#retrieving-the-latest-code
[issue]: https://github.com/scikit-learn/scikit-learn/issues/8688
[setup-fork]: https://docs.scipy.org/doc/numpy/dev/gitwash/development_setup.html#forking
[feature-branch]: https://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html#making-a-new-feature-branch