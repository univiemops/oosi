# -*- coding: utf-8 -*-
"""Scikit-learn compatible custom splitters
v013

@author: david.steyrl@gmail.com
"""

import itertools
import math
import numpy as np
import pandas as pd
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
    """Randomized group shuffle split cross-validator with optional non-replacement sampling,
    predefined training groups, test sample limits, and unused-sample reinjection.

    Provides randomized train/test splits based on groups. Allows sampling of
    test set group combinations with or without replacement. When sampling
    without replacement, duplicate test set group combinations are prevented until
    all possible combinations have been exhausted.

    Optionally, predefined training groups (`train_groups`) can be specified.
    When provided, the baseline training set consists exclusively of samples belonging to
    these `train_groups`.

    Testing sample counts per group can be capped via `max_test_samples_per_group`. When
    set, remaining unused test samples per group can be partially re-injected into the
    training set using `n_inject_samples` via two strategy choices:
    1. `'random'`: Randomly selects `n_inject_samples` per group from unused test samples
       and appends them to the training set.
    2. `'lowest_target'`: Calculates the mean target value ($y$) per group of a specified
       feature column in $X$ over the current baseline training set. Then, selects the
       `n_inject_samples` per test group with the lowest mean target value from the remaining
       unused testing pool and injects them back into the training set.

    Parameters
    ----------
    n_splits : int, default=5
        Number of re-shuffling & splitting iterations.

    n_test_groups : int or float, default=1
        If int, represents the absolute number of candidate groups to include in the test set.
        If float, represents the fraction of available candidate groups (excluding `train_groups`)
        to include in the test set.

    train_groups : array-like or None, default=None
        Group labels that should be explicitly fixed to constitute the training set.
        These groups are excluded from candidate testing groups. If None, the baseline training
        set dynamically consists of all groups not selected for the test set.

    max_test_samples_per_group : int or None, default=None
        Maximum number of samples to randomly draw from each selected testing group for the final
        test set.

    n_inject_samples : int or None, default=None
        Number of unused test samples per group to reinject into the training set. Must be <=
        total unused test samples available per group.

    injection_strategy : {'random', 'lowest_target'}, default='random'
        Strategy used to pick samples from the unused test pool to reinject into the training set:
        - `'random'`: Randomly pick `n_inject_samples` per selected test group.
        - `'lowest_target'`: Select samples corresponding to the lowest average target ($y$)
          computed over the training set using the grouping column specified in `feature_group_col`.

    feature_group_col : int, str, or None, default=None
        Column index or name in $X$ representing the grouping variable used to compute mean target values
        when `injection_strategy='lowest_target'`.

    replace : bool, default=False
        Whether to sample test set group combinations with replacement. If False,
        avoids repeated test group combinations across splits.

    max_combinations_exhaustive : int, default=10000
        Threshold for switching between exhaustive combination generation and
        sample-and-reject sampling when `replace=False`.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the split generation.

    Examples
    --------
    Standard usage without reinjection:

    >>> import numpy as np
    >>> from custom_splitter import ExtendedGroupShuffleSplit
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> groups = np.array([1, 1, 2, 2, 3, 3])
    >>> egss = ExtendedGroupShuffleSplit(n_splits=3, n_test_groups=1, replace=False, random_state=42)
    >>> for train_idx, test_idx in egss.split(X, y, groups):
    ...     print(f"TRAIN: {train_idx} TEST: {test_idx}")
    TRAIN: [0 1 4 5] TEST: [2 3]
    TRAIN: [2 3 4 5] TEST: [0 1]
    TRAIN: [0 1 2 3] TEST: [4 5]

    Reinjecting unused test samples using random strategy:

    >>> groups = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    >>> X = np.zeros((12, 2))
    >>> y = np.arange(12)
    >>> egss_inject = ExtendedGroupShuffleSplit(
    ...     n_splits=1, n_test_groups=1, train_groups=[1],
    ...     max_test_samples_per_group=1, n_inject_samples=2,
    ...     injection_strategy='random', random_state=42
    ... )
    >>> for train_idx, test_idx in egss_inject.split(X, y, groups):
    ...     print(f"TRAIN: {train_idx} TEST: {test_idx}")
    TRAIN: [0 1 2 3 4 5] TEST: [6]

    Reinjecting unused test samples using lowest average target strategy:

    >>> X = np.array([[0], [0], [1], [1], [0], [0], [1], [1], [0], [0], [1], [1]])
    >>> y = np.array([10, 10, 2, 2, 5, 5, 1, 1, 8, 8, 3, 3])
    >>> egss_target = ExtendedGroupShuffleSplit(
    ...     n_splits=1, n_test_groups=1, train_groups=[1],
    ...     max_test_samples_per_group=1, n_inject_samples=1,
    ...     injection_strategy='lowest_target', feature_group_col=0, random_state=42
    ... )
    >>> for train_idx, test_idx in egss_target.split(X, y, groups):
    ...     print(f"TRAIN: {train_idx} TEST: {test_idx}")
    TRAIN: [0 1 2 3 6] TEST: [4]
    """

    def __init__(
        self,
        n_splits=5,
        *,
        n_test_groups=1,
        train_groups=None,
        max_test_samples_per_group=None,
        n_inject_samples=None,
        injection_strategy="random",
        feature_group_col=None,
        replace=False,
        max_combinations_exhaustive=10000,
        random_state=None,
    ):
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.train_groups = train_groups
        self.max_test_samples_per_group = max_test_samples_per_group
        self.n_inject_samples = n_inject_samples
        self.injection_strategy = injection_strategy
        self.feature_group_col = feature_group_col
        self.replace = replace
        self.max_combinations_exhaustive = max_combinations_exhaustive
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits

    def _subsample_test_indices(self, groups, selected_test_groups, rng):
        """Filters test set indices per selected group and identifies unused test indices."""
        test_indices = []
        unused_test_indices_per_group = {}

        for grp in selected_test_groups:
            grp_indices = np.where(groups == grp)[0]
            if (
                self.max_test_samples_per_group is not None
                and len(grp_indices) > self.max_test_samples_per_group
            ):
                selected = rng.choice(
                    grp_indices, size=self.max_test_samples_per_group, replace=False
                )
                unused = np.setdiff1d(grp_indices, selected)
            else:
                selected = grp_indices
                unused = np.array([], dtype=int)

            test_indices.extend(selected)
            unused_test_indices_per_group[grp] = unused

        return np.sort(np.array(test_indices, dtype=int)), unused_test_indices_per_group

    def _select_injected_indices(self, X, y, base_train_idx, unused_test_indices_per_group, rng):
        """Selects indices to inject back into the training set according to specified strategy."""
        if self.n_inject_samples is None or self.n_inject_samples <= 0:
            return np.array([], dtype=int)

        injected_indices = []

        if self.injection_strategy == "random":
            for grp, unused_idx in unused_test_indices_per_group.items():
                if len(unused_idx) == 0:
                    continue
                n_select = min(self.n_inject_samples, len(unused_idx))
                chosen = rng.choice(unused_idx, size=n_select, replace=False)
                injected_indices.extend(chosen)

        elif self.injection_strategy == "lowest_target":
            if y is None:
                raise ValueError("Target 'y' must be provided when using 'lowest_target' injection strategy.")
            if X is None or self.feature_group_col is None:
                raise ValueError(
                    "Predictor 'X' and 'feature_group_col' must be provided when using 'lowest_target' strategy."
                )

            # Extract grouping column from X and ensure 1D
            if hasattr(X, "iloc"):
                if isinstance(self.feature_group_col, str):
                    feat_groups = X[self.feature_group_col].to_numpy()
                else:
                    feat_groups = X.iloc[:, self.feature_group_col].to_numpy()
            else:
                feat_groups = np.asarray(X)[:, self.feature_group_col]

            # Flatten to 1D arrays to prevent pandas 2D array ValueError
            feat_groups = np.asarray(feat_groups).ravel()
            y_arr = np.asarray(y).ravel()

            # Compute average target per feature group on current training set
            train_feat_groups = feat_groups[base_train_idx]
            train_y = y_arr[base_train_idx]

            df_train = pd.DataFrame({"feat_grp": train_feat_groups, "y": train_y})
            mean_y_per_feat_grp = df_train.groupby("feat_grp")["y"].mean().to_dict()

            overall_mean_y = df_train["y"].mean()

            # Select n samples with lowest average target per test group
            for grp, unused_idx in unused_test_indices_per_group.items():
                if len(unused_idx) == 0:
                    continue

                unused_feat_grps = feat_groups[unused_idx]
                unused_means = np.array(
                    [mean_y_per_feat_grp.get(fg, overall_mean_y) for fg in unused_feat_grps]
                )

                # Sort unused indices by average target ascending
                sorted_positions = np.argsort(unused_means)
                n_select = min(self.n_inject_samples, len(unused_idx))
                chosen = unused_idx[sorted_positions[:n_select]]
                injected_indices.extend(chosen)
        else:
            raise ValueError(f"Unknown injection strategy: {self.injection_strategy}")

        return np.array(injected_indices, dtype=int)

    def _iter_test_indices(self, X, y, groups):
        groups = _validate_groups(groups)
        unique_groups = np.unique(groups)

        if self.train_groups is not None:
            fixed_train_groups = np.unique(self.train_groups)
            candidate_test_groups = np.setdiff1d(unique_groups, fixed_train_groups)
        else:
            fixed_train_groups = None
            candidate_test_groups = unique_groups

        n_candidate_groups = len(candidate_test_groups)

        if n_candidate_groups == 0:
            raise ValueError(
                "No candidate groups remaining for testing. Check 'train_groups'."
            )

        if isinstance(self.n_test_groups, float):
            if not 0.0 < self.n_test_groups < 1.0:
                raise ValueError("n_test_groups as float must be between 0.0 and 1.0.")
            k_test_groups = int(np.ceil(self.n_test_groups * n_candidate_groups))
        elif isinstance(self.n_test_groups, int):
            k_test_groups = self.n_test_groups
        else:
            raise TypeError("n_test_groups must be an int or float.")

        if fixed_train_groups is None and k_test_groups >= n_candidate_groups:
            raise ValueError(
                f"n_test_groups={k_test_groups} must be < total groups ({n_candidate_groups}) "
                "when train_groups is None."
            )
        if k_test_groups < 1 or k_test_groups > n_candidate_groups:
            raise ValueError(
                f"n_test_groups={k_test_groups} must be >= 1 and <= available candidate test groups ({n_candidate_groups})."
            )

        if (
            self.max_test_samples_per_group is not None
            and self.max_test_samples_per_group < 1
        ):
            raise ValueError("max_test_samples_per_group must be >= 1 or None.")

        rng = check_random_state(self.random_state)
        total_combos = math.comb(n_candidate_groups, k_test_groups)

        if self.replace:
            for _ in range(self.n_splits):
                selected_test_groups = rng.choice(
                    candidate_test_groups, size=k_test_groups, replace=False
                )
                yield self._subsample_test_indices(groups, selected_test_groups, rng)
        else:
            if total_combos <= self.max_combinations_exhaustive:
                # Exhaustive Approach
                all_combinations = list(
                    itertools.combinations(candidate_test_groups, k_test_groups)
                )
                rng.shuffle(all_combinations)

                for i in range(self.n_splits):
                    combo = all_combinations[i % total_combos]
                    yield self._subsample_test_indices(groups, combo, rng)
            else:
                # Sample-and-Reject Approach
                seen_combinations = set()
                for _ in range(self.n_splits):
                    if len(seen_combinations) == total_combos:
                        seen_combinations.clear()

                    selected_test_groups = tuple(
                        sorted(
                            rng.choice(
                                candidate_test_groups,
                                size=k_test_groups,
                                replace=False,
                            )
                        )
                    )
                    while selected_test_groups in seen_combinations:
                        selected_test_groups = tuple(
                            sorted(
                                rng.choice(
                                    candidate_test_groups,
                                    size=k_test_groups,
                                    replace=False,
                                )
                            )
                        )

                    seen_combinations.add(selected_test_groups)
                    yield self._subsample_test_indices(
                        groups, selected_test_groups, rng
                    )

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,), default=None
            Target values.

        groups : array-like of shape (n_samples,)
            Group labels for the samples.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        groups = _validate_groups(groups)
        rng = check_random_state(self.random_state)

        for test_idx, unused_test_indices_per_group in self._iter_test_indices(
            X, y, groups
        ):
            if self.train_groups is not None:
                fixed_train_groups = np.unique(self.train_groups)
                base_train_idx = np.where(np.isin(groups, fixed_train_groups))[0]
            else:
                assigned_test_groups = list(unused_test_indices_per_group.keys())
                base_train_idx = np.where(~np.isin(groups, assigned_test_groups))[0]

            injected_idx = self._select_injected_indices(
                X, y, base_train_idx, unused_test_indices_per_group, rng
            )

            final_train_idx = np.sort(np.concatenate([base_train_idx, injected_idx]))
            yield final_train_idx, test_idx
