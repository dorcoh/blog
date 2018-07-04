---
layout: post
title:  "Tutorial: Contribute to sklearn, or other open-source projects"
date:   2018-07-03 00:00:00 +0200
categories: python machine-learning github opensource 
comments: true
---

Lately I've been interested in understanding more deeply some of the modules in `sklearn` library and decided the best way to learn is to contribute to the open-source project, which also helps the community. The post goal then is to make the contribution process easier for newcomers, we will examine and fix together a bug.

<b>Prerequisites:</b>
You should be familiar with <i>Python</i>, <i>Git</i> and have a Github account

# Get started

To get started log into your Github account and fork the latest repository of [scikit-learn][sklearn-git], then clone it to your directory of choice using this command (you can retrieve the link with the green button which says 'Clone or download'):

`git clone https://github.com/your_user_name/scikit-learn.git`

This should take a while, next change directory and set an upstream remote (a link to the main repository):

```
cd scikit-learn
git remote add upstream https://github.com/scikit-learn/scikit-learn.git
```

By typing `git remote -v show` you should notice there are two remotes, one from your Github account cloned repository, and the other from scikit:

```python
origin  https://github.com/dorcoh/scikit-learn.git (fetch)
origin  https://github.com/dorcoh/scikit-learn.git (push)
upstream    https://github.com/scikit-learn/scikit-learn.git (fetch)
upstream    https://github.com/scikit-learn/scikit-learn.git (push)
```

Then you can sync your local repository with scikit's using:

 `git fetch upstream`

Instead of the usual `git pull` which will sync with origin (though you can configure git to pull automatically from upstream, see references for further details). 

And when you start working on a new feature you should initialize a branch by:

`git checkout -b my-new-feature upstream/master`

# Virtual Environment

For better work flow you should use virtual environments, basically they help us manage our dependencies and avoid conflicts. There are few choices available, I use here Python's `virtualenv`. To install it type:

`pip install virtualenv`

And then, configure a new environment by:

`virtualenv -p /usr/bin/python3 env_scikit`

This will install the environment requirements locally on your chosen directory (I chose `env_scikit` but you can pick any directory name you want). Activate the environment using `source env_scikit/bin/activate`, when you finish working exit by `deactivate`.

While it's activated, if you run Python interpreter or install new packages by `pip install` it will perform only from the environment configured directory.

# Install dependencies and external modules

Before you start hacking you should install scikit's library dependencies, there are couple of methods suggested in their documentation (e.g., setuptools) but the preferred way is using pip: 

`pip install --editable .` 

Each time source code of compiled extension changes (yours or pulled from upstream) type: 

`python setup.py build_ext --inplace`

On Unix based systems you could call `make` on main directory to build and launch all tests (notice this could easily take an hour, though you should look into the `Makefile`)

* for more details see scikit's contribution page on references.

## Make changes

# Explore and understand the bug

From here on I'll describe a conrecte example, an actual bug that I'd like to fix. Specifically, there's a [bug][issue] with `CountVectorizer`.

Before we dive into fixing this, I'll give you a usage example for `CountVectorizer`, for those of you who aren't familiar with the feature extraction module.

<h3>CountVectorizer usage example</h3>

```python
In [1]: from sklearn.feature_extraction import text

In [2]: cv = text.CountVectorizer()

In [3]: res = cv.fit_transform(["good news everyone, everyone good!"])

In [4]: res.toarray()
Out[4]: array([[2, 2, 1]], dtype=int64)

In [5]: cv.get_feature_names()
Out[5]: ['everyone', 'good', 'news']
```
As you can see, this call counted our words occurences (after it has done some preprocessing, such as to_lower and removing punctuation marks), it is very common to perform this on NLP tasks. In this example I chose 1-grams, we can also choose to count 1-grams and 2-grams:

```python
In [6]: cv = text.CountVectorizer(ngram_range=(1,2))

In [7]: res = cv.fit_transform(["good news everyone, everyone good!"])

In [8]: res.toarray()
Out[8]: array([[2, 1, 1, 2, 1, 1, 1]], dtype=int64)

In [9]: cv.get_feature_names()
Out[9]: 
['everyone',
 'everyone everyone',
 'everyone good',
 'good',
 'good news',
 'news',
 'news everyone']
```

As you noticed this class has a property `ngram_range` which allows to choose range for extracting n-grams (e.g., 1 <= n <= 2 extracts 1-gram and 2-grams), the problem is a missing validation for this parameter, it turns out that we could set `ngram_range=(2,1)`, which doesn't make sense of course, and more importantly it could break other stuff that assumes its validity. Following `sklearn` current guidelines, we should perform the validation on `fit(..)` function rather than `__init__(..)` - the class constructor. 

Let's explore the corresponding module `feature_extraction/text.py`, take a look at these class definitions:

```python
class HashingVectorizer(BaseEstimator, VectorizerMixin, TransformerMixin)
class CountVectorizer(BaseEstimator, VectorizerMixin)
class TfidfVectorizer(CountVectorizer)
```

All of them implement `fit(..)` and has the `ngram_range` property, however `TfidfVectorizer`, as well as its `fit(..)`  is inherited from `CountVectorizer`, making our life a bit easier.

# Add test

The next step is deciding on the expected behaviour when encountering this bug. We can reproduce the bug using a custom script, or Python console.

However, the more standard way is to write a unit test (`sklearn` uses `PyTest` for this purpose - you can install it through `pip`), which should fail at this moment. Test scripts are usually stored in `our_module/tests` directory, let's run tests for `feature_extraction` module:

```python
(env_scikit) d@de:~/scikit-learn/sklearn/feature_extraction$ pytest tests/test_text.py 
==================================================== test session starts ======================================================
platform linux -- Python 3.5.1+, pytest-3.5.1, py-1.5.3, pluggy-0.6.0
rootdir: /home/deebee/personal/opensource/scikit-learn, inifile: setup.cfg
collected 45 items                                                                                                                                                                                         

tests/test_text.py .............................................                                                                                                                                     [100%]

============================================== 45 passed, 2 warnings in 1.09 seconds =========================================
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
 
<b>Explanation:</b> First, we use `@pytest.mark.parametrize` decorator to make this a parametric test (or else we'll have to create an instance for each of the tested classes). Then I initialize the objects with faulty `ngram_range`, and check if it raises the corresponding error and message while calling `fit` or `fit_transform` (notice only `HashingVectorizer` implements `transform` function).

When running the test again, it should fail (I skipped some of the output for readability):

```python
(scikit) deebee@realm:~/Desktop/open-source/scikit-learn/sklearn/feature_extraction$ pytest tests/test_text.py 
.
.
.
../utils/testing.py:402: AssertionError
============================== 1 failed, 45 passed, 2 warnings in 2.61 seconds ==============================
```

You can read more about unit tests in Python docs, or this wonderful [blog][hitchhiker].

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

Then I called it in relevant functions of each class, I describe here only one of the modified functions which is `HashingVectorizer.fit(..)`:

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

# What's next?

Assuming you've tested your module, you should also run all other tests to validate you haven't broken anything else (not always necessary but a good practice): type `make test` in main directory , it should take a while (could be an hour or so).

Specifically for `scikit` they follow coding standards (see on their contributing page), then you should also check for PEP8/pyflakes errors.

Next push changes and [submit a pull request][pull-request], then wait for code review.

Hope you enjoyed this post, happy comitting!


# References

* [Sklearn git repository][sklearn-git]
* [Sklearn contribution page][sklearn-contribute]
* [Numpy setting development workflow guide][numpy-devel]
* [Astropy development guide][astropy-devel]
* [Pipenv & Virtual enviroments - The hitchhiker's guide to Python blog][hitchhiker-pipenv]
* [Python unit tests - The hitchhiker's guide to Python blog][unit-tests]

[sklearn-git]: https://github.com/scikit-learn/scikit-learn
[sklearn-contribute]: http://scikit-learn.org/stable/developers/contributing.html
[numpy-devel]: https://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html
[astropy-devel]: http://astropy.readthedocs.io/en/latest/development/workflow/development_workflow.html
[issue]: https://github.com/scikit-learn/scikit-learn/issues/8688
[setup-fork]: https://docs.scipy.org/doc/numpy/dev/gitwash/development_setup.html#forking
[feature-branch]: https://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html#making-a-new-feature-branch
[pull-request]: https://help.github.com/articles/creating-a-pull-request-from-a-fork
[unit-tests]: http://docs.python-guide.org/en/latest/writing/tests/
[hitchhiker-pipenv]: http://docs.python-guide.org/en/latest/dev/virtualenvs/
[hitchhiker]: http://docs.python-guide.org/en/latest/