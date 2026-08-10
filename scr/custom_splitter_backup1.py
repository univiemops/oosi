# -*- coding: utf-8 -*-
"""Scikit-learn compatible custom splitters
v011

@author: david.steyrl@gmail.com
"""

import math
from itertools import combinations
import numpy as np
from sklearn.utils.validation import check_array, check_random_state
from sklearn.model_selection._split import (
    _BaseKFold,
    _RepeatedSplits,
)
from sklearn.model_selection import GroupShuffleSplit


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
    """K-fold iterator variant with non-overlapping groups[cite: 2].

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
    """Repeated Group K-Fold cross validator[cite: 2].

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


class TargetSortedGroupShuffleSplit(GroupShuffleSplit):
    """Group Shuffle Split with test sub-sampling and optional target-sorted training injection[cite: 2].

    Provides randomized train/test splits based on groups with extended control[cite: 2]
    over sample size retention within groups and the option to inject unused test
    samples back into the training split sorted by stimulus target means[cite: 2].

    Parameters
    ----------
    n_splits : int, default=5
        Number of splitting iterations[cite: 2].

    test_size : float, int, or None, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of groups to include in the test split. If int, represents the
        absolute number of groups[cite: 2].

    train_size : float, int, or None, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of groups to include in the train split. If int, represents the
        absolute number of groups[cite: 2].

    n_test_samples : int or "all", default="all"
        Number of samples per group kept in final test set[cite: 2].

    n_train_samples_back : int, default=0
        Number of unused test samples sent back into train set[cite: 2].

    sort_by_target : bool, default=False
        Whether to sort injected unused test samples by stimulus target mean[cite: 2].

    stim_id_col : str, int, or None, default="stim_id"
        Column identifier for stimulus IDs (used when `sort_by_target=True`)[cite: 2].

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the split[cite: 2].

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from custom_splitter import TargetSortedGroupShuffleSplit
    >>> df = pd.DataFrame({
    ...     "stim_id": [10, 10, 20, 20, 30, 30, 40, 40],
    ...     "feature": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ... })
    >>> y = np.array([1, 2, 1, 3, 2, 4, 1, 5])
    >>> groups = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    >>> splitter = TargetSortedGroupShuffleSplit(
    ...     n_splits=2,
    ...     test_size=2,
    ...     n_test_samples=1,
    ...     n_train_samples_back=1,
    ...     random_state=42
    ... )
    >>> for train_idx, test_idx in splitter.split(df, y, groups):
    ...     print(f"TRAIN: {train_idx} TEST: {test_idx}")
    TRAIN: [0 1 2 3 6] TEST: [4 7]
    TRAIN: [2 3 4 5 1] TEST: [0 6]
    """

    def __init__(
        self,
        n_splits=5,
        *,
        test_size=None,
        train_size=None,
        n_test_samples="all",
        n_train_samples_back=0,
        sort_by_target=False,
        stim_id_col="stim_id",
        random_state=None,
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self.n_test_samples = n_test_samples
        self.n_train_samples_back = n_train_samples_back
        self.sort_by_target = sort_by_target
        self.stim_id_col = stim_id_col

    def _extract_stim_ids(self, X):
        """Extract stim_id array from pandas DataFrame or 2D numpy array X[cite: 2].

        Parameters
        ----------
        X : {array-like, pandas.DataFrame} of shape (n_samples, n_features)
            Data array containing stimulus identification values.

        Returns
        -------
        stim_ids : ndarray of shape (n_samples,)
            Extracted 1D array of stimulus IDs.

        Raises
        ------
        ValueError
            If `X` is a numpy array without column names and `stim_id_col` is a string.
        """
        if hasattr(X, "columns") and self.stim_id_col in X.columns:
            res = np.asarray(X[self.stim_id_col])
        elif hasattr(X, "iloc"):
            if isinstance(self.stim_id_col, int):
                res = np.asarray(X.iloc[:, self.stim_id_col])
            else:
                res = np.asarray(X[self.stim_id_col])
        else:
            X_arr = np.asarray(X)
            if isinstance(self.stim_id_col, int):
                res = X_arr[:, self.stim_id_col]
            elif isinstance(self.stim_id_col, str):
                raise ValueError(
                    f"X is a numpy array without column names, but stim_id_col='{self.stim_id_col}' "
                    "was provided as a string. Please pass an integer column index."
                )
            else:
                res = X_arr[:, self.stim_id_col]
        return np.asarray(res).ravel()

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set[cite: 2].

        Parameters
        ----------
        X : {array-like, pandas.DataFrame} of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems. Required if
            `sort_by_target=True`[cite: 2].

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set[cite: 2].

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        groups = _validate_groups(groups).ravel()

        if self.sort_by_target:
            if y is None:
                raise ValueError(
                    "Target variable 'y' must be provided when sort_by_target=True."
                )
            y = np.asarray(y).ravel()
            stim_ids = self._extract_stim_ids(X)

        if not (isinstance(self.n_test_samples, int) or self.n_test_samples == "all"):
            raise ValueError("n_test_samples must be an integer or 'all'.")
        if isinstance(self.n_test_samples, int) and self.n_test_samples < 1:
            raise ValueError("n_test_samples must be >= 1.")
        if not isinstance(self.n_train_samples_back, int):
            raise ValueError("n_train_samples_back must be an integer.")
        if self.n_train_samples_back < 0:
            raise ValueError("n_train_samples_back must be >= 0.")

        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        test_size = self.test_size
        train_size = self.train_size

        if test_size is None and train_size is None:
            test_size = 0.2

        if test_size is not None:
            if isinstance(test_size, (int, np.integer)):
                if test_size >= n_groups or test_size <= 0:
                    raise ValueError(
                        f"test_size={test_size} should be a positive integer and "
                        f"smaller than the number of unique groups ({n_groups})."
                    )
                n_test = int(test_size)
            else:
                if test_size <= 0.0 or test_size >= 1.0:
                    raise ValueError(
                        f"test_size={test_size} should be a float in the interval (0.0, 1.0)."
                    )
                n_test = int(np.floor(test_size * n_groups))
                if n_test == 0:
                    n_test = 1
        else:
            n_test = None

        if train_size is not None:
            if isinstance(train_size, (int, np.integer)):
                if train_size >= n_groups or train_size <= 0:
                    raise ValueError(
                        f"train_size={train_size} should be a positive integer and "
                        f"smaller than the number of unique groups ({n_groups})."
                    )
                n_train = int(train_size)
            else:
                if train_size <= 0.0 or train_size >= 1.0:
                    raise ValueError(
                        f"train_size={train_size} should be a float in the interval (0.0, 1.0)."
                    )
                n_train = int(np.floor(train_size * n_groups))
                if n_train == 0:
                    n_train = 1
        else:
            n_train = None

        if n_test is None:
            n_test = n_groups - n_train
        elif n_train is None:
            n_train = n_groups - n_test

        if n_test + n_train > n_groups:
            raise ValueError(
                f"The sum of train_size and test_size ({n_test + n_train}) "
                f"is greater than the number of unique groups ({n_groups})."
            )

        rng = check_random_state(self.random_state)
        C = math.comb(n_groups, n_test)

        selected_combos = []
        if C <= 10000:
            all_combos = list(combinations(sorted(unique_groups), n_test))
            rng.shuffle(all_combos)

            for i in range(self.n_splits):
                selected_combos.append(all_combos[i % C])
        else:
            used = set()
            target_unique = min(self.n_splits, C)

            while len(selected_combos) < target_unique:
                combo = tuple(
                    sorted(rng.choice(unique_groups, size=n_test, replace=False))
                )
                if combo not in used:
                    used.add(combo)
                    selected_combos.append(combo)

            if self.n_splits > len(selected_combos):
                extended = []
                for i in range(self.n_splits):
                    extended.append(selected_combos[i % len(selected_combos)])
                selected_combos = extended

        for split_idx in range(self.n_splits):
            test_groups_combo = selected_combos[split_idx]
            test_groups_set = set(test_groups_combo)

            remaining_groups = np.array(
                [g for g in unique_groups if g not in test_groups_set]
            )

            if n_train < len(remaining_groups):
                train_groups_combo = rng.choice(
                    remaining_groups, size=n_train, replace=False
                )
            else:
                train_groups_combo = remaining_groups

            train_index = np.where(np.isin(groups, train_groups_combo))[0].ravel()
            test_index = np.where(np.isin(groups, test_groups_combo))[0].ravel()

            if self.n_test_samples == "all":
                yield train_index, test_index
                continue

            unique_test_groups = np.unique(groups[test_index])

            new_test_indices = []
            unused_test_indices = []

            for g in unique_test_groups:
                g_indices = np.where(groups == g)[0]
                shuffled_g_indices = rng.permutation(g_indices)

                n_total = len(shuffled_g_indices)
                n_test_sel = min(self.n_test_samples, n_total)

                g_test = shuffled_g_indices[:n_test_sel]
                new_test_indices.extend(g_test)

                g_unused = shuffled_g_indices[n_test_sel:]
                unused_test_indices.extend(g_unused)

            if self.n_train_samples_back > 0 and len(unused_test_indices) > 0:
                if self.sort_by_target:
                    train_stims = stim_ids[train_index]
                    train_y = y[train_index]

                    stim_target_sums = {}
                    stim_target_counts = {}
                    for s, target in zip(train_stims, train_y):
                        stim_target_sums[s] = stim_target_sums.get(s, 0.0) + float(target)
                        stim_target_counts[s] = stim_target_counts.get(s, 0) + 1

                    stim_mean_dict = {
                        s: stim_target_sums[s] / stim_target_counts[s]
                        for s in stim_target_sums
                    }

                    unused_arr = np.array(unused_test_indices, dtype=int).ravel()
                    unused_stims = stim_ids[unused_arr]

                    unused_avg_targets = np.array(
                        [stim_mean_dict.get(s, float("inf")) for s in unused_stims],
                        dtype=float,
                    )

                    sorted_order = np.argsort(unused_avg_targets, kind="stable")
                    sorted_unused_indices = unused_arr[sorted_order]

                    n_back = min(self.n_train_samples_back, len(sorted_unused_indices))
                    back_indices = sorted_unused_indices[:n_back].ravel()
                else:
                    n_back = min(self.n_train_samples_back, len(unused_test_indices))
                    back_indices = np.array(unused_test_indices[:n_back], dtype=int).ravel()

                final_train_index = np.concatenate([train_index, back_indices])
                final_train_index.sort()
            else:
                final_train_index = train_index.copy()

            final_test_index = np.array(new_test_indices, dtype=int).ravel()
            final_test_index.sort()

            yield final_train_index, final_test_index