# -*- coding: utf-8 -*-
"""Scikit-learn compatible custom splitters
v010

@author: david.steyrl@gmail.com
"""

import math
from itertools import combinations
import numpy as np
from sklearn.utils.validation import check_array, check_random_state
from sklearn.model_selection._split import _BaseKFold, _RepeatedSplits, BaseCrossValidator
from sklearn.model_selection import GroupShuffleSplit


def _validate_groups(groups):
    """Helper function to validate the groups parameter across splitters."""
    if groups is None:
        raise ValueError("The 'groups' parameter should not be None.")
    return check_array(groups, ensure_2d=False, dtype=None)


class GroupKFold(_BaseKFold):
    """
    K-fold iterator variant with non-overlapping groups.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    The folds are approximately balanced in the sense that the number of
    distinct groups is approximately the same in each fold.
    Read more in the :ref:`User Guide <group_k_fold>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
        Note that the samples within each split will not be shuffled.
    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold. Otherwise, this
        parameter has no effect.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from custom_splitter import GroupKFold
    >>> X = np.array([[10], [20], [30], [40], [50], [60], [70], [80], [90], [100], [110], [120], [130], [140], [150], [160], [170]])
    >>> y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    >>> groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    >>> group_kfold = GroupKFold(n_splits=2, shuffle=True, random_state=3141592)
    >>> for i, (train_index, test_index) in enumerate(group_kfold.split(X, y, groups)):
    ...    print(f"Fold {i}:")
    ...    print(f"  Train: index={train_index}, group={groups[train_index]}")
    ...    print(f"  Test:  index={test_index}, group={groups[test_index]}")
    ...
    Fold 0:
      Train: index=[ 6  7  8 13 14 15 16], group=[2 2 2 4 4 4 4]
      Test:  index=[ 0  1  2  3  4  5  9 10 11 12], group=[0 0 0 1 1 1 3 3 3 3]
    Fold 1:
      Train: index=[ 0  1  2  3  4  5  9 10 11 12], group=[0 0 0 1 1 1 3 3 3 3]
      Test:  index=[ 6  7  8 13 14 15 16], group=[2 2 2 4 4 4 4]
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

        # Weight groups by their number of occurrences
        n_samples_per_group = np.bincount(groups.squeeze())

        # Distribute the most frequent groups first
        indices = np.argsort(n_samples_per_group)[::-1]

        if self.shuffle:
            rng = check_random_state(self.random_state)
            for n_sample in np.unique(n_samples_per_group):
                same_n_indices_index = np.where(n_samples_per_group == n_sample)[0]
                target_chunk = indices[same_n_indices_index]
                rng.shuffle(target_chunk)
                indices[same_n_indices_index] = target_chunk

        n_samples_per_group = n_samples_per_group[indices]

        # Total weight of each fold
        n_samples_per_fold = np.zeros(self.n_splits)

        # Mapping from group index to fold index
        group_to_fold = np.zeros(len(unique_groups))

        # Distribute samples by adding the largest weight to the lightest fold
        for group_index, weight in enumerate(n_samples_per_group):
            lightest_fold = np.argmin(n_samples_per_fold)
            n_samples_per_fold[lightest_fold] += weight
            group_to_fold[indices[group_index]] = lightest_fold

        indices = group_to_fold[groups]

        for f in range(self.n_splits):
            yield np.where(indices == f)[0]

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
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
    """
    Repeated Group K-Fold cross validator. Repeats Group K-Fold n times with
    different randomization in each repetition.
    Read more in the :ref:`User Guide <repeated_group_k_fold>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of each repeated cross-validation instance.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from custom_splitter import RepeatedGroupKFold
    >>> X = np.array([[10], [20], [30], [40], [50], [60], [70], [80], [90], [100], [110], [120], [130], [140], [150], [160], [170]])
    >>> y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    >>> groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    >>> rkf = RepeatedGroupKFold(n_splits=2, n_repeats=2, random_state=3141592)
    >>> for i, (train_index, test_index) in enumerate(rkf.split(X, y, groups)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}, group={groups[train_index]}")
    ...     print(f"  Test:  index={test_index}, group={groups[test_index]}")
    ...
    Fold 0:
      Train: index=[ 6  7  8 13 14 15 16], group=[2 2 2 4 4 4 4]
      Test:  index=[ 0  1  2  3  4  5  9 10 11 12], group=[0 0 0 1 1 1 3 3 3 3]
    Fold 1:
      Train: index=[ 0  1  2  3  4  5  9 10 11 12], group=[0 0 0 1 1 1 3 3 3 3]
      Test:  index=[ 6  7  8 13 14 15 16], group=[2 2 2 4 4 4 4]
    Fold 2:
      Train: index=[ 0  1  2 13 14 15 16], group=[0 0 0 4 4 4 4]
      Test:  index=[ 3  4  5  6  7  8  9 10 11 12], group=[1 1 1 2 2 2 3 3 3 3]
    Fold 3:
      Train: index=[ 3  4  5  6  7  8  9 10 11 12], group=[1 1 1 2 2 2 3 3 3 3]
      Test:  index=[ 0  1  2 13 14 15 16], group=[0 0 0 4 4 4 4]
    """

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(
            GroupKFold,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
        )

    def split(self, X, y=None, groups=None):
        """
        Generates indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
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


class CustomGroupShuffleSplit(GroupShuffleSplit):
    """
    Group Shuffle Split with non-repeating group assignments in the test set,
    sub-sampling of test set groups, and training set injection.

    This splitter partitions the data by assigning combinations of unique groups 
    to either the train or test set. To ensure balanced coverage of groups and prevent 
    redundancies, it enforces that there are no repeating combinations of groups assigned 
    to the test set across splits until the maximum possible combinations are exhausted.

    Once test groups are chosen, it sub-samples them keeping only `n_test_samples`
    per group in the final test set. Optionally, a specified number of the unused/leftover
    samples can be sent back into the training set (`n_train_samples_back`).

    Parameters
    ----------
    n_splits : int, default=5
        Number of splitting iterations. Enforces no duplicate test group sets across
        these splits up to the unique combinations limit.
    test_size : float, int, or None, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of groups to include in the test split. If int, represents the
        absolute number of test groups. If None, the value is set to 0.2.
    train_size : float, int, or None, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of groups to include in the train split. If int,
        represents the absolute number of train groups. If None, the value
        is automatically set to the complement of the test size.
    n_test_samples : int or "all", default="all"
        The number of randomly drawn samples per group of the groups in the test set
        that are kept in the final test set. If "all", all group samples are kept.
    n_train_samples_back : int, default=0
        The number of samples among the unused/leftover samples (per test group)
        to send back into the training set.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of group partitioning and within-group sub-sampling.

    Examples
    --------
    >>> import numpy as np
    >>> from custom_splitter import CustomGroupShuffleSplit
    >>> X = np.array([[10], [20], [30], [40], [50], [60], [70], [80], [90], [100], [110], [120], [130], [140], [150], [160], [170]])
    >>> y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    >>> groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    >>> 
    >>> # 1. Test case: fallback to leave one group out
    >>> gss = CustomGroupShuffleSplit(n_splits=5, test_size=1, n_test_samples="all", random_state=3141592)
    >>> for i, (train_index, test_index) in enumerate(gss.split(X, y, groups)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}, group={groups[train_index]}")
    ...     print(f"  Test:  index={test_index}, group={groups[test_index]}")
    Fold 0:
      Train: index=[ 0  1  2  6  7  8  9 10 11 12 13 14 15 16], group=[0 0 0 2 2 2 3 3 3 3 4 4 4 4]
      Test:  index=[3 4 5], group=[1 1 1]
    Fold 1:
      Train: index=[ 0  1  2  3  4  5  6  7  8 13 14 15 16], group=[0 0 0 1 1 1 2 2 2 4 4 4 4]
      Test:  index=[ 9 10 11 12], group=[3 3 3 3]
    Fold 2:
      Train: index=[ 0  1  2  3  4  5  9 10 11 12 13 14 15 16], group=[0 0 0 1 1 1 3 3 3 3 4 4 4 4]
      Test:  index=[6 7 8], group=[2 2 2]
    Fold 3:
      Train: index=[ 0  1  2  3  4  5  6  7  8  9 10 11 12], group=[0 0 0 1 1 1 2 2 2 3 3 3 3]
      Test:  index=[13 14 15 16], group=[4 4 4 4]
    Fold 4:
      Train: index=[ 3  4  5  6  7  8  9 10 11 12 13 14 15 16], group=[1 1 1 2 2 2 3 3 3 3 4 4 4 4]
      Test:  index=[0 1 2], group=[0 0 0]
    >>> 
    >>> # 2. Test case: fallback to leave p groups out
    >>> gss_all = CustomGroupShuffleSplit(n_splits=3, test_size=2, n_test_samples="all", random_state=3141592)
    >>> for i, (train_index, test_index) in enumerate(gss_all.split(X, y, groups)):
    ...    print(f"Fold {i}:")
    ...    print(f"  Train: index={train_index}, group={groups[train_index]}")
    ...    print(f"  Test:  index={test_index}, group={groups[test_index]}")
    Fold 0:
      Train: index=[ 0  1  2  3  4  5 13 14 15 16], group=[0 0 0 1 1 1 4 4 4 4]
      Test:  index=[ 6  7  8  9 10 11 12], group=[2 2 2 3 3 3 3]
    Fold 1:
      Train: index=[ 3  4  5  9 10 11 12 13 14 15 16], group=[1 1 1 3 3 3 3 4 4 4 4]
      Test:  index=[0 1 2 6 7 8], group=[0 0 0 2 2 2]
    Fold 2:
      Train: index=[ 0  1  2  6  7  8 13 14 15 16], group=[0 0 0 2 2 2 4 4 4 4]
      Test:  index=[ 3  4  5  9 10 11 12], group=[1 1 1 3 3 3 3]
    >>> 
    >>> # 3. Test case: only two random samples per group in test set
    >>> gss_sub = CustomGroupShuffleSplit(n_splits=3, test_size=2, n_test_samples=2, n_train_samples_back=0, random_state=3141592)
    >>> for i, (train_index, test_index) in enumerate(gss_sub.split(X, y, groups)):
    ...    print(f"Fold {i}:")
    ...    print(f"  Train: index={train_index}, group={groups[train_index]}")
    ...    print(f"  Test:  index={test_index}, group={groups[test_index]}")
    Fold 0:
      Train: index=[ 0  1  2  3  4  5 13 14 15 16], group=[0 0 0 1 1 1 4 4 4 4]
      Test:  index=[ 6  7 11 12], group=[2 2 3 3]
    Fold 1:
      Train: index=[ 3  4  5  9 10 11 12 13 14 15 16], group=[1 1 1 3 3 3 3 4 4 4 4]
      Test:  index=[0 1 6 7], group=[0 0 2 2]
    Fold 2:
      Train: index=[ 0  1  2  6  7  8 13 14 15 16], group=[0 0 0 2 2 2 4 4 4 4]
      Test:  index=[ 3  4 10 12], group=[1 1 3 3]
    >>> 
    >>> # 4. Test case: only two random samples per group in test set and up to two samples send back to training set
    >>> gss_inject = CustomGroupShuffleSplit(n_splits=2, test_size=2, n_test_samples=2, n_train_samples_back=2, random_state=3141592)
    >>> for i, (train_index, test_index) in enumerate(gss_inject.split(X, y, groups)):
    ...    print(f"Fold {i}:")
    ...    print(f"  Train: index={train_index}, group={groups[train_index]}")
    ...    print(f"  Test:  index={test_index}, group={groups[test_index]}")
    Fold 0:
      Train: index=[ 0  1  2  3  4  5  8  9 10 13 14 15 16], group=[0 0 0 1 1 1 2 3 3 4 4 4 4]
      Test:  index=[ 6  7 11 12], group=[2 2 3 3]
    Fold 1:
      Train: index=[ 2  3  4  5  8  9 10 11 12 13 14 15 16], group=[0 1 1 1 2 3 3 3 3 4 4 4 4]
      Test:  index=[0 1 6 7], group=[0 0 2 2]
    """

    def __init__(
        self,
        n_splits=5,
        *,
        test_size=None,
        train_size=None,
        n_test_samples="all",
        n_train_samples_back=0,
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

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
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
        groups = _validate_groups(groups)
        
        # Parameter validation
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

        # Parse test_size and train_size with respect to group counts
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
            else:  # float
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
            else:  # float
                if train_size <= 0.0 or train_size >= 1.0:
                    raise ValueError(
                        f"train_size={train_size} should be a float in the interval (0.0, 1.0)."
                    )
                n_train = int(np.floor(train_size * n_groups))
                if n_train == 0:
                    n_train = 1
        else:
            n_train = None

        # Resolve None parameter complements
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
        
        # Calculate maximum possible combinations C = n_groups choose n_test
        C = math.comb(n_groups, n_test)
        
        selected_combos = []
        if C <= 10000:
            # Generate all possible test group combinations, shuffle, and take first n_splits
            all_combos = list(combinations(sorted(unique_groups), n_test))
            rng.shuffle(all_combos)
            
            for i in range(self.n_splits):
                selected_combos.append(all_combos[i % C])
        else:
            # Rejection sampling for unique combinations to prevent memory overflow
            used = set()
            target_unique = min(self.n_splits, C)
            
            while len(selected_combos) < target_unique:
                combo = tuple(sorted(rng.choice(unique_groups, size=n_test, replace=False)))
                if combo not in used:
                    used.add(combo)
                    selected_combos.append(combo)
            
            # If splits exceed maximum unique combinations, cycle them
            if self.n_splits > len(selected_combos):
                extended = []
                for i in range(self.n_splits):
                    extended.append(selected_combos[i % len(selected_combos)])
                selected_combos = extended

        # Yield indices for each partitioned split combination
        for split_idx in range(self.n_splits):
            test_groups_combo = selected_combos[split_idx]
            test_groups_set = set(test_groups_combo)
            
            # Remaining groups available for train set
            remaining_groups = np.array([g for g in unique_groups if g not in test_groups_set])
            
            if n_train < len(remaining_groups):
                train_groups_combo = rng.choice(remaining_groups, size=n_train, replace=False)
            else:
                train_groups_combo = remaining_groups

            # Extract base indices belonging to the selected groups
            train_index = np.where(np.isin(groups, train_groups_combo))[0]
            test_index = np.where(np.isin(groups, test_groups_combo))[0]

            if self.n_test_samples == "all":
                yield train_index, test_index
                continue

            # Identify the unique groups assigned to the test set for sub-sampling
            unique_test_groups = np.unique(groups[test_index])

            new_test_indices = []
            new_train_back_indices = []

            for g in unique_test_groups:
                g_indices = np.where(groups == g)[0]
                shuffled_g_indices = rng.permutation(g_indices)
                
                n_total = len(shuffled_g_indices)
                n_test_sel = min(self.n_test_samples, n_total)

                # Keep a random subset of size n_test_sel for the test set
                g_test = shuffled_g_indices[:n_test_sel]
                new_test_indices.extend(g_test)

                # Collect leftover samples
                g_unused = shuffled_g_indices[n_test_sel:]

                # Inject up to n_train_samples_back back into train set
                if self.n_train_samples_back > 0 and len(g_unused) > 0:
                    n_back = min(self.n_train_samples_back, len(g_unused))
                    g_back = g_unused[:n_back]
                    new_train_back_indices.extend(g_back)

            if new_train_back_indices:
                final_train_index = np.concatenate([train_index, new_train_back_indices])
                final_train_index.sort()
            else:
                final_train_index = train_index.copy()

            final_test_index = np.array(new_test_indices)
            final_test_index.sort()

            yield final_train_index, final_test_index