# -*- coding: utf-8 -*-
"""Scikit-learn compatible custom splitters
v013

@author: david.steyrl@gmail.com
"""

import itertools
import math
import numpy as np
from sklearn.utils.validation import check_array, check_random_state
from sklearn.model_selection._split import _BaseKFold, _RepeatedSplits


def _validate_groups(groups):
    """Validate the groups array.

    Parameters
    ----------
    groups : array-like of shape (n_samples,)
        Group labels for the samples used while splitting the dataset into
        train/test set.

    Returns
    -------
    groups : ndarray
        The checked and validated groups array.

    Raises
    ------
    ValueError
        If `groups` is None.

    Examples
    --------
    >>> import numpy as np
    >>> from custom_splitter import _validate_groups
    >>> groups = [1, 1, 2, 2]
    >>> _validate_groups(groups)
    array([1, 1, 2, 2])
    """
    if groups is None:
        raise ValueError("The 'groups' parameter should not be None.")
    return check_array(groups, ensure_2d=False, dtype=None)


class GroupKFold(_BaseKFold):
    """K-fold iterator variant with non-overlapping groups.

    Provides train/test indices to split data according to a third-party
    provided group. This group information can be used to encode arbitrary
    domain-specific stratifications (e.g. patient, subject, experimenters).

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    shuffle : bool, default=False
        Whether to shuffle group order prior to assigning groups to folds.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness when `shuffle=True`. Pass an int for
        reproducible output across multiple function calls.

    Examples
    --------
    >>> import numpy as np
    >>> from custom_splitter import GroupKFold
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> groups = np.array([1, 1, 2, 2, 3, 3])
    >>> gkf = GroupKFold(n_splits=3)
    >>> for train_idx, test_idx in gkf.split(X, y, groups):
    ...     print(f"TRAIN: {train_idx} TEST: {test_idx}")
    TRAIN: [0 1 2 3] TEST: [4 5]
    TRAIN: [0 1 4 5] TEST: [2 3]
    TRAIN: [2 3 4 5] TEST: [0 1]
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_indices(self, X, y, groups):
        groups = _validate_groups(groups)
        unique_groups, groups = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError(
                "Cannot have number of splits n_splits=%d greater"
                " than the number of groups: %d." % (self.n_splits, n_groups)
            )

        n_samples_per_group = np.bincount(groups.squeeze())
        indices = np.argsort(n_samples_per_group)[::-1]

        if self.shuffle:
            rng = check_random_state(self.random_state)
            for n_sample in np.unique(n_samples_per_group):
                same_n_indices_index = np.where(n_samples_per_group == n_sample)[0]
                target_chunk = indices[same_n_indices_index]
                rng.shuffle(target_chunk)
                indices[same_n_indices_index] = target_chunk

        n_samples_per_group = n_samples_per_group[indices]
        n_samples_per_fold = np.zeros(self.n_splits)
        group_to_fold = np.zeros(len(unique_groups))

        for group_index, weight in enumerate(n_samples_per_group):
            lightest_fold = np.argmin(n_samples_per_fold)
            n_samples_per_fold[lightest_fold] += weight
            group_to_fold[indices[group_index]] = lightest_fold

        indices = group_to_fold[groups]

        for f in range(self.n_splits):
            yield np.where(indices == f)[0]

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        return super().split(X, y, groups)


class RepeatedGroupKFold(_RepeatedSplits):
    """Repeated Group K-Fold cross validator.

    Repeats Group K-Fold n times with different randomization in each repetition.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    n_repeats : int, default=10
        Number of times cross-validator should be repeated.

    random_state : int, RandomState instance or None, default=None
        Controls the generation of random states for each repetition. Pass an int
        for reproducible output across multiple function calls.

    Examples
    --------
    >>> import numpy as np
    >>> from custom_splitter import RepeatedGroupKFold
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> groups = np.array([1, 1, 2, 2, 3, 3])
    >>> rgkf = RepeatedGroupKFold(n_splits=3, n_repeats=2, random_state=42)
    >>> for train_idx, test_idx in rgkf.split(X, y, groups):
    ...     print(f"TRAIN: {train_idx} TEST: {test_idx}")
    TRAIN: [0 1 2 3] TEST: [4 5]
    TRAIN: [0 1 4 5] TEST: [2 3]
    TRAIN: [2 3 4 5] TEST: [0 1]
    TRAIN: [0 1 4 5] TEST: [2 3]
    TRAIN: [2 3 4 5] TEST: [0 1]
    TRAIN: [0 1 2 3] TEST: [4 5]
    """

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(
            GroupKFold,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
        )

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        n_repeats = self.n_repeats
        rng = check_random_state(self.random_state)

        for idx in range(n_repeats):
            cv = self.cv(random_state=rng, shuffle=True, **self.cvargs)
            for train_index, test_index in cv.split(X, y, groups):
                yield train_index, test_index


class ExtendedGroupShuffleSplit:
    """Randomized train/test splits based on groups with sampling and injection control.

    Provides randomized train/test splits to divide data according to
    third-party provided group labels. Extends scikit-learn's standard GroupShuffleSplit
    by controlling test group combination sampling with or without replacement.
    Additionally allows predefining training groups, limiting maximum test samples
    per group, and injecting unused test samples back into the training set.

    Parameters
    ----------
    n_splits : int, default=5
        Number of re-shuffling & splitting iterations.

    n_test_groups : int or float, default=1
        If int, represents the absolute number of groups to include in the test set.
        If float, represents the fraction of total unique groups to include in the test set.

    replace : bool, default=False
        Whether to sample test set group combinations with replacement.
        If False, duplicate test group combinations are not repeated across splits.

    train_groups : array-like or None, default=None
        Predefined array of group labels that are fixed to the training set.
        These groups will never be sampled for the test set.

    max_test_samples_per_group : int or None, default=None
        Maximum number of testing samples per group in the testing set. If set, after
        finding the testing groups, only up to `max_test_samples_per_group` randomly 
        drawn samples per test group are included in the test set. The remaining samples 
        from those test groups are omitted from both train and test sets, unless back-injection 
        is enabled.

    n_inject_per_group : int or None, default=None
        Number of omitted/unused test samples per group to inject back into the training set.
        Requires `max_test_samples_per_group` to be set.

    inject_strategy : {'random', 'target_mean'} or None, default=None
        Strategy used to select which unused test samples to inject back into training:
        - 'random': Randomly select `n_inject_per_group` samples per group from the unused samples.
        - 'target_mean': Compute the mean target (`y`) per group in `X[:, inject_group_col]` 
          over the initial training set, then select the `n_inject_per_group` samples 
          belonging to the groups with the lowest mean target values from the unused samples.

    inject_group_col : int, str or None, default=None
        Column index or column name in `X` specifying the grouping variable used when 
        `inject_strategy='target_mean'` or when grouping by a specific column in `X`.

    exhaustive_threshold : int, default=10000
        Maximum number of total potential group combinations allowed to trigger the
        exhaustive sampling mode when `replace=False`.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of sampling. Pass an int for reproducible output.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> X = pd.DataFrame({
    ...     'feat1': [1, 1, 1, 1, 2, 2, 2, 2],
    ...     'grp_col': [10, 10, 10, 10, 20, 20, 20, 20]
    ... })
    >>> y = np.array([10, 20, 30, 40, 1, 2, 3, 4])
    >>> groups = np.array([1, 1, 1, 1, 2, 2, 2, 2])

    >>> egss_rand = ExtendedGroupShuffleSplit(
    ...     n_splits=1,
    ...     n_test_groups=1,
    ...     max_test_samples_per_group=1,
    ...     n_inject_per_group=2,
    ...     inject_strategy='random',
    ...     random_state=42
    ... )
    >>> for train_idx, test_idx in egss_rand.split(X, y, groups):
    ...     print(f"TRAIN: {train_idx} TEST: {test_idx}")
    """

    def __init__(
        self,
        n_splits=5,
        *,
        n_test_groups=1,
        replace=False,
        train_groups=None,
        max_test_samples_per_group=None,
        n_inject_per_group=None,
        inject_strategy=None,
        inject_group_col=None,
        exhaustive_threshold=10000,
        random_state=None,
    ):
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.replace = replace
        self.train_groups = train_groups
        self.max_test_samples_per_group = max_test_samples_per_group
        self.n_inject_per_group = n_inject_per_group
        self.inject_strategy = inject_strategy
        self.inject_group_col = inject_group_col
        self.exhaustive_threshold = exhaustive_threshold
        self.random_state = random_state

    def _clean_inject_group_col(self):
        """Sanitize YAML string inputs for inject_group_col."""
        if self.inject_group_col in (None, "None", "none", "NONE"):
            return None
        return self.inject_group_col

    def _get_n_test_groups(self, n_groups):
        if isinstance(self.n_test_groups, float):
            if not 0.0 < self.n_test_groups < 1.0:
                raise ValueError(
                    f"n_test_groups={self.n_test_groups} should be a float in (0, 1)."
                )
            n_test = math.ceil(self.n_test_groups * n_groups)
        elif isinstance(self.n_test_groups, int):
            n_test = self.n_test_groups
        else:
            raise TypeError("n_test_groups must be an int or a float.")

        if n_test >= n_groups or n_test <= 0:
            raise ValueError(
                f"n_test_groups={n_test} must be strictly greater than 0 and "
                f"less than the total number of groups ({n_groups})."
            )
        return n_test

    def _subsample_test_indices(self, groups, test_groups, rng):
        """Sample test indices up to max_test_samples_per_group per test group."""
        if self.max_test_samples_per_group is None:
            test_indices = np.isin(groups, test_groups).nonzero()[0]
            return test_indices, np.array([], dtype=int)

        if not isinstance(self.max_test_samples_per_group, int) or self.max_test_samples_per_group <= 0:
            raise ValueError("max_test_samples_per_group must be a positive integer or None.")

        selected_test_indices = []
        unused_test_indices = []

        for tg in test_groups:
            tg_indices = np.where(groups == tg)[0]
            if len(tg_indices) > self.max_test_samples_per_group:
                sampled = rng.choice(
                    tg_indices, size=self.max_test_samples_per_group, replace=False
                )
                unused = np.setdiff1d(tg_indices, sampled)
                selected_test_indices.extend(sampled)
                unused_test_indices.extend(unused)
            else:
                selected_test_indices.extend(tg_indices)

        return (
            np.array(sorted(selected_test_indices), dtype=int),
            np.array(sorted(unused_test_indices), dtype=int),
        )

    def _clean_train_groups(self):
        """Sanitize string inputs or empty structures for train_groups."""
        if self.train_groups in (None, "None", "none", "NONE"):
            return None
        return self.train_groups

    def _iter_test_indices(self, X, y, groups):
        groups = _validate_groups(groups)
        unique_groups = np.unique(groups)

        train_groups = self._clean_train_groups()
        if train_groups is not None:
            predefined_train = np.unique(check_array(train_groups, ensure_2d=False, dtype=None))
            eligible_test_groups = np.setdiff1d(unique_groups, predefined_train)
        else:
            eligible_test_groups = unique_groups

        n_groups = len(eligible_test_groups)
        if n_groups == 0:
            raise ValueError("No remaining groups available for test set split after excluding train_groups.")

        n_test = self._get_n_test_groups(n_groups)
        rng = check_random_state(self.random_state)
        total_combinations = math.comb(n_groups, n_test)

        if self.replace:
            for _ in range(self.n_splits):
                test_groups = rng.choice(eligible_test_groups, size=n_test, replace=False)
                yield self._subsample_test_indices(groups, test_groups, rng)
        else:
            if total_combinations <= self.exhaustive_threshold:
                all_combos = list(itertools.combinations(eligible_test_groups, n_test))
                rng.shuffle(all_combos)
                for i in range(self.n_splits):
                    combo = all_combos[i % total_combinations]
                    yield self._subsample_test_indices(groups, combo, rng)
            else:
                seen_combos = set()
                for _ in range(self.n_splits):
                    if len(seen_combos) == total_combinations:
                        seen_combos.clear()

                    while True:
                        test_groups = tuple(
                            sorted(rng.choice(eligible_test_groups, size=n_test, replace=False))
                        )
                        if test_groups not in seen_combos:
                            seen_combos.add(test_groups)
                            break
                    
                    yield self._subsample_test_indices(groups, test_groups, rng)

    def _extract_column(self, X, col):
        """Extract a 1D numpy array for a specific column regardless of input format."""
        if hasattr(X, "iloc"):  # Pandas DataFrame / Series
            if isinstance(col, str):
                val = X[col].to_numpy()
            else:
                val = X.iloc[:, col].to_numpy()
        else:  # NumPy Array
            val = np.asarray(X)[:, col]
        
        return np.asarray(val).ravel()

    def _clean_inject_strategy(self):
        """Sanitize strategy inputs to support None, 'None', 'none', 'random', 'target_mean'."""
        if self.inject_strategy in (None, "None", "none", "NONE"):
            return None
        return str(self.inject_strategy).lower()

    def _select_injections(self, X, y, groups, initial_train_idx, unused_test_idx, rng):
        """Select which unused test samples to inject back into training."""
        strategy = self._clean_inject_strategy()

        # Return empty array if strategy is None or no unused test indices exist
        if strategy is None or self.n_inject_per_group is None or len(unused_test_idx) == 0:
            return np.array([], dtype=int)

        inject_col = self._clean_inject_group_col()

        # Extract group labels as a guaranteed 1D numpy array
        if inject_col is not None:
            unused_group_labels = self._extract_column(X, inject_col)[unused_test_idx]
        else:
            unused_group_labels = np.asarray(groups)[unused_test_idx].ravel()

        unique_unused_groups = np.unique(unused_group_labels)

        if strategy == "random":
            injected_indices = []
            for g in unique_unused_groups:
                g_mask = (unused_group_labels == g).ravel()
                g_indices = unused_test_idx[g_mask]

                n_select = min(len(g_indices), self.n_inject_per_group)
                if n_select > 0:
                    chosen = rng.choice(g_indices, size=n_select, replace=False)
                    injected_indices.extend(chosen)

            return np.array(injected_indices, dtype=int)

        elif strategy == "target_mean":
            if inject_col is None:
                raise ValueError("`inject_group_col` must be specified when using `inject_strategy='target_mean'`.")
            if y is None:
                raise ValueError("`y` cannot be None when using `inject_strategy='target_mean'`.")

            y_arr = np.asarray(y).ravel()
            train_group_labels = self._extract_column(X, inject_col)[initial_train_idx]
            y_train = y_arr[initial_train_idx]

            # Compute average target (y) per group in training set
            unique_train_groups = np.unique(train_group_labels)
            group_means = {
                g: y_train[train_group_labels == g].mean() for g in unique_train_groups
            }

            injected_indices = []
            for g in unique_unused_groups:
                g_mask = (unused_group_labels == g).ravel()
                g_unused_indices = unused_test_idx[g_mask]

                n_select = min(len(g_unused_indices), self.n_inject_per_group)
                if n_select > 0:
                    if g in group_means:
                        sorted_order = np.argsort(y_arr[g_unused_indices])
                        injected_indices.extend(g_unused_indices[sorted_order[:n_select]])
                    else:
                        injected_indices.extend(rng.choice(g_unused_indices, size=n_select, replace=False))

            return np.array(injected_indices, dtype=int)

        else:
            raise ValueError(
                f"Unknown inject_strategy: '{self.inject_strategy}'. Choice of None, 'random', or 'target_mean'."
            )

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set."""
        groups_validated = _validate_groups(groups)
        rng = check_random_state(self.random_state)

        for test_index, unused_test_index in self._iter_test_indices(X, y, groups_validated):
            test_groups = np.unique(groups_validated[test_index])
            initial_train_index = np.where(~np.isin(groups_validated, test_groups))[0]

            if self.inject_strategy is not None and len(unused_test_index) > 0:
                injected_index = self._select_injections(
                    X, y, groups_validated, initial_train_index, unused_test_index, rng
                )
                train_index = np.sort(np.concatenate([initial_train_index, injected_index]))
            else:
                train_index = initial_train_index

            yield train_index, test_index

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits