# -*- coding: utf-8 -*-
"""Unit tests for custom_splitter.py."""

import sys
import numpy as np
import pandas as pd
import pytest

sys.path.append(r"../scr/")

from custom_splitter import (
    GroupKFold,
    RepeatedGroupKFold,
    TargetSortedGroupShuffleSplit,
    _validate_groups,
)

# =============================================================================
# Helper Function Tests: _validate_groups
# =============================================================================


def test_validate_groups_valid():
    """Test _validate_groups with valid input list/array."""
    groups = [1, 1, 2, 2, 3]
    validated = _validate_groups(groups)
    assert isinstance(validated, np.ndarray)
    np.testing.assert_array_equal(validated, np.array([1, 1, 2, 2, 3]))


def test_validate_groups_none_raises():
    """Test _validate_groups raises ValueError when groups is None."""
    with pytest.raises(ValueError, match="The 'groups' parameter should not be None."):
        _validate_groups(None)


# =============================================================================
# GroupKFold Tests
# =============================================================================


class TestGroupKFold:
    @pytest.fixture
    def sample_data(self):
        X = np.arange(12).reshape(6, 2)
        y = np.array([1, 2, 3, 4, 5, 6])
        groups = np.array([1, 1, 2, 2, 3, 3])
        return X, y, groups

    def test_basic_split_count_and_non_overlapping_groups(self, sample_data):
        X, y, groups = sample_data
        gkf = GroupKFold(n_splits=3)
        splits = list(gkf.split(X, y, groups))

        assert len(splits) == 3

        all_test_indices = []
        for train_idx, test_idx in splits:
            # Validate non-overlapping group constraint
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            assert train_groups.isdisjoint(test_groups)
            all_test_indices.extend(test_idx)

        # Validate that every sample is tested exactly once
        assert sorted(all_test_indices) == list(range(len(X)))

    def test_shuffle_reproducibility(self, sample_data):
        X, y, groups = sample_data
        gkf1 = GroupKFold(n_splits=3, shuffle=True, random_state=42)
        gkf2 = GroupKFold(n_splits=3, shuffle=True, random_state=42)

        splits1 = list(gkf1.split(X, y, groups))
        splits2 = list(gkf2.split(X, y, groups))

        for (tr1, te1), (tr2, te2) in zip(splits1, splits2):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)

    def test_n_splits_greater_than_n_groups_raises(self, sample_data):
        X, y, groups = sample_data
        gkf = GroupKFold(n_splits=5)  # only 3 unique groups
        with pytest.raises(
            ValueError, match="Cannot have number of splits n_splits=5 greater than"
        ):
            list(gkf.split(X, y, groups))


# =============================================================================
# RepeatedGroupKFold Tests
# =============================================================================


class TestRepeatedGroupKFold:
    def test_repeated_splits_count(self):
        X = np.arange(20).reshape(10, 2)
        y = np.arange(10)
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

        n_splits = 5
        n_repeats = 3
        rgkf = RepeatedGroupKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=42
        )

        splits = list(rgkf.split(X, y, groups))
        assert len(splits) == n_splits * n_repeats

    def test_repeat_reproducibility(self):
        X = np.arange(12).reshape(6, 2)
        y = np.arange(6)
        groups = np.array([1, 1, 2, 2, 3, 3])

        rgkf1 = RepeatedGroupKFold(n_splits=3, n_repeats=2, random_state=123)
        rgkf2 = RepeatedGroupKFold(n_splits=3, n_repeats=2, random_state=123)

        splits1 = list(rgkf1.split(X, y, groups))
        splits2 = list(rgkf2.split(X, y, groups))

        for (tr1, te1), (tr2, te2) in zip(splits1, splits2):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)


# =============================================================================
# TargetSortedGroupShuffleSplit Tests
# =============================================================================


class TestTargetSortedGroupShuffleSplit:
    @pytest.fixture
    def setup_dataframe_data(self):
        df = pd.DataFrame(
            {
                "stim_id": [10, 10, 20, 20, 30, 30, 40, 40, 50, 50],
                "feature1": np.linspace(0, 1, 10),
                "feature2": np.linspace(10, 20, 10),
            }
        )
        y = np.array([1.0, 3.0, 2.0, 4.0, 5.0, 1.0, 3.0, 3.0, 2.0, 6.0])
        groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        return df, y, groups

    def test_stim_id_extraction_pandas_dataframe_by_name(self, setup_dataframe_data):
        df, y, groups = setup_dataframe_data
        splitter = TargetSortedGroupShuffleSplit(stim_id_col="stim_id")
        extracted = splitter._extract_stim_ids(df)
        np.testing.assert_array_equal(extracted, df["stim_id"].to_numpy())

    def test_stim_id_extraction_pandas_dataframe_by_int_index(
        self, setup_dataframe_data
    ):
        df, y, groups = setup_dataframe_data
        splitter = TargetSortedGroupShuffleSplit(stim_id_col=0)
        extracted = splitter._extract_stim_ids(df)
        np.testing.assert_array_equal(extracted, df["stim_id"].to_numpy())

    def test_stim_id_extraction_numpy_array_by_int_index(self):
        X_arr = np.array([[10, 0.1], [10, 0.2], [20, 0.3], [20, 0.4]])
        splitter = TargetSortedGroupShuffleSplit(stim_id_col=0)
        extracted = splitter._extract_stim_ids(X_arr)
        np.testing.assert_array_equal(extracted, np.array([10, 10, 20, 20]))

    def test_stim_id_extraction_numpy_array_string_col_raises(self):
        X_arr = np.array([[10, 0.1], [10, 0.2]])
        splitter = TargetSortedGroupShuffleSplit(stim_id_col="stim_id")
        with pytest.raises(ValueError, match="X is a numpy array without column names"):
            splitter._extract_stim_ids(X_arr)

    def test_basic_group_shuffle_default(self, setup_dataframe_data):
        df, y, groups = setup_dataframe_data
        splitter = TargetSortedGroupShuffleSplit(
            n_splits=3, test_size=0.4, random_state=42
        )
        splits = list(splitter.split(df, y, groups))

        assert len(splits) == 3
        for train_idx, test_idx in splits:
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            assert train_groups.isdisjoint(test_groups)

    def test_fixed_train_groups_enforcement(self, setup_dataframe_data):
        df, y, groups = setup_dataframe_data
        fixed_groups = [1, 2]
        splitter = TargetSortedGroupShuffleSplit(
            n_splits=3,
            test_size=1,
            fixed_train_groups=fixed_groups,
            random_state=42,
        )
        splits = list(splitter.split(df, y, groups))

        for train_idx, test_idx in splits:
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])

            # Fixed groups must strictly be present in training and absent from testing
            for fg in fixed_groups:
                assert fg in train_groups
                assert fg not in test_groups

    def test_integer_test_and_train_sizes(self, setup_dataframe_data):
        df, y, groups = setup_dataframe_data
        splitter = TargetSortedGroupShuffleSplit(
            n_splits=2, test_size=2, train_size=2, random_state=42
        )
        splits = list(splitter.split(df, y, groups))

        for train_idx, test_idx in splits:
            assert len(np.unique(groups[train_idx])) == 2
            assert len(np.unique(groups[test_idx])) == 2

    def test_subsampling_test_samples(self, setup_dataframe_data):
        df, y, groups = setup_dataframe_data
        splitter = TargetSortedGroupShuffleSplit(
            n_splits=2, test_size=2, n_test_samples=1, random_state=42
        )
        splits = list(splitter.split(df, y, groups))

        for train_idx, test_idx in splits:
            test_groups = groups[test_idx]
            # Each test group should only contribute 1 sample
            unique, counts = np.unique(test_groups, return_counts=True)
            assert all(c == 1 for c in counts)

    def test_train_samples_back_injection(self, setup_dataframe_data):
        df, y, groups = setup_dataframe_data
        splitter = TargetSortedGroupShuffleSplit(
            n_splits=1,
            test_size=2,
            n_test_samples=1,
            n_train_samples_back=1,
            sort_by_target=False,
            random_state=42,
        )
        splits = list(splitter.split(df, y, groups))
        train_idx, test_idx = splits[0]

        # 3 groups in train (6 samples) + 1 back-injected unused test sample = 7 samples
        assert len(train_idx) == 7
        assert len(test_idx) == 2

    def test_target_sorted_train_samples_back(self, setup_dataframe_data):
        df, y, groups = setup_dataframe_data
        splitter = TargetSortedGroupShuffleSplit(
            n_splits=1,
            test_size=2,
            n_test_samples=1,
            n_train_samples_back=1,
            sort_by_target=True,
            stim_id_col="stim_id",
            random_state=42,
        )
        splits = list(splitter.split(df, y, groups))
        assert len(splits) == 1
        train_idx, test_idx = splits[0]
        assert len(train_idx) == 7
        assert len(test_idx) == 2

    def test_combinatorial_sampling_small_vs_large_combinations(self):
        # Test small combinations (<= 10,000)
        X = np.arange(20).reshape(10, 2)
        groups = np.repeat(np.arange(5), 2)
        splitter_small = TargetSortedGroupShuffleSplit(
            n_splits=10, test_size=2, random_state=42
        )
        splits_small = list(splitter_small.split(X, groups=groups))
        assert len(splits_small) == 10

        # Test large combinations (> 10,000 combinations)
        large_groups = np.repeat(np.arange(30), 2)
        large_X = np.arange(120).reshape(60, 2)
        splitter_large = TargetSortedGroupShuffleSplit(
            n_splits=5, test_size=10, random_state=42
        )
        splits_large = list(splitter_large.split(large_X, groups=large_groups))
        assert len(splits_large) == 5

    # --- Exception Validation Tests ---

    def test_no_remaining_pool_groups_raises(self, setup_dataframe_data):
        df, y, groups = setup_dataframe_data
        # Assign all unique groups (1 to 5) as fixed train groups
        all_groups = list(np.unique(groups))
        splitter = TargetSortedGroupShuffleSplit(fixed_train_groups=all_groups)
        with pytest.raises(
            ValueError, match="No eligible groups remaining for the test split pool."
        ):
            list(splitter.split(df, y, groups))

    def test_sort_by_target_missing_y_raises(self, setup_dataframe_data):
        df, y, groups = setup_dataframe_data
        splitter = TargetSortedGroupShuffleSplit(sort_by_target=True)
        with pytest.raises(
            ValueError,
            match="Target variable 'y' must be provided when sort_by_target=True",
        ):
            list(splitter.split(df, y=None, groups=groups))

    def test_invalid_n_test_samples_raises(self, setup_dataframe_data):
        df, y, groups = setup_dataframe_data
        with pytest.raises(
            ValueError, match="n_test_samples must be an integer or 'all'"
        ):
            splitter = TargetSortedGroupShuffleSplit(n_test_samples="invalid")
            list(splitter.split(df, y, groups))

        with pytest.raises(ValueError, match="n_test_samples must be >= 1"):
            splitter = TargetSortedGroupShuffleSplit(n_test_samples=0)
            list(splitter.split(df, y, groups))

    def test_invalid_n_train_samples_back_raises(self, setup_dataframe_data):
        df, y, groups = setup_dataframe_data
        with pytest.raises(ValueError, match="n_train_samples_back must be an integer"):
            splitter = TargetSortedGroupShuffleSplit(n_train_samples_back="invalid")
            list(splitter.split(df, y, groups))

        with pytest.raises(ValueError, match="n_train_samples_back must be >= 0"):
            splitter = TargetSortedGroupShuffleSplit(n_train_samples_back=-1)
            list(splitter.split(df, y, groups))

    def test_invalid_test_size_raises(self, setup_dataframe_data):
        df, y, groups = setup_dataframe_data
        # Out of bounds int
        with pytest.raises(
            ValueError, match="should be a positive integer and smaller than"
        ):
            splitter = TargetSortedGroupShuffleSplit(test_size=10)
            list(splitter.split(df, y, groups))

        # Out of bounds float
        with pytest.raises(ValueError, match="should be a float in the interval"):
            splitter = TargetSortedGroupShuffleSplit(test_size=1.5)
            list(splitter.split(df, y, groups))

    def test_invalid_train_size_raises(self, setup_dataframe_data):
        df, y, groups = setup_dataframe_data
        with pytest.raises(
            ValueError, match="should be a positive integer and smaller than"
        ):
            splitter = TargetSortedGroupShuffleSplit(train_size=10)
            list(splitter.split(df, y, groups))

        with pytest.raises(ValueError, match="should be a float in the interval"):
            splitter = TargetSortedGroupShuffleSplit(train_size=1.5)
            list(splitter.split(df, y, groups))

    def test_train_and_test_size_sum_exceeds_groups_raises(self, setup_dataframe_data):
        df, y, groups = setup_dataframe_data
        with pytest.raises(ValueError, match="The sum of train_size and test_size"):
            splitter = TargetSortedGroupShuffleSplit(train_size=3, test_size=3)
            list(splitter.split(df, y, groups))


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))