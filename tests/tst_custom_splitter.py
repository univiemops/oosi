# -*- coding: utf-8 -*-
"""unit tests for custom_splitter.py
v001

@author: david.steyrl@gmail.com
"""

import numpy as np
import pandas as pd
import sys
import unittest

sys.path.append("../scr/")

# Import classes and helper directly from the custom module
# Replace 'custom_splitter' with your module filename if different.
from custom_splitter import (
    _validate_groups,
    GroupKFold,
    RepeatedGroupKFold,
    ExtendedGroupShuffleSplit,
)


class TestValidateGroups(unittest.TestCase):
    """Tests for the helper function `_validate_groups`."""

    def test_valid_groups(self):
        groups = [1, 1, 2, 2, 3]
        validated = _validate_groups(groups)
        self.assertIsInstance(validated, np.ndarray)
        np.testing.assert_array_equal(validated, np.array(groups))

    def test_none_groups_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            _validate_groups(None)
        self.assertIn("parameter should not be None", str(ctx.exception))


class TestGroupKFold(unittest.TestCase):
    """Tests for `GroupKFold` cross-validator."""

    def setUp(self):
        self.X = np.ones((12, 2))
        self.y = np.arange(12)
        self.groups = np.array([1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5])

    def test_basic_split_folds(self):
        gkf = GroupKFold(n_splits=3)
        splits = list(gkf.split(self.X, self.y, self.groups))
        
        self.assertEqual(len(splits), 3)
        
        all_test_indices = []
        for train_idx, test_idx in splits:
            # Check train and test index overlap
            self.assertEqual(len(np.intersect1d(train_idx, test_idx)), 0)
            # Check that all indices total the dataset size
            self.assertEqual(len(train_idx) + len(test_idx), len(self.X))
            all_test_indices.extend(test_idx)
            
        # Ensure test sets cover all indices without overlap
        self.assertEqual(sorted(all_test_indices), list(range(len(self.X))))

    def test_no_group_leakage(self):
        gkf = GroupKFold(n_splits=3)
        for train_idx, test_idx in gkf.split(self.X, self.y, self.groups):
            train_groups = set(self.groups[train_idx])
            test_groups = set(self.groups[test_idx])
            self.assertTrue(train_groups.isdisjoint(test_groups))

    def test_shuffle_reproducibility(self):
        gkf1 = GroupKFold(n_splits=3, shuffle=True, random_state=42)
        gkf2 = GroupKFold(n_splits=3, shuffle=True, random_state=42)

        splits1 = list(gkf1.split(self.X, self.y, self.groups))
        splits2 = list(gkf2.split(self.X, self.y, self.groups))

        for (tr1, te1), (tr2, te2) in zip(splits1, splits2):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)

    def test_nsplits_greater_than_ngroups_raises(self):
        gkf = GroupKFold(n_splits=10)
        # 12 samples to match self.X and self.y, but only 2 unique groups (1 and 2)
        groups_12_samples = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
        
        with self.assertRaises(ValueError) as ctx:
            list(gkf.split(self.X, self.y, groups=groups_12_samples))
            
        self.assertIn("greater than the number of groups", str(ctx.exception))


class TestRepeatedGroupKFold(unittest.TestCase):
    """Tests for `RepeatedGroupKFold` cross-validator."""

    def setUp(self):
        self.X = np.ones((10, 2))
        self.y = np.arange(10)
        self.groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    def test_repeated_splits_count(self):
        n_splits, n_repeats = 3, 4
        rgkf = RepeatedGroupKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
        splits = list(rgkf.split(self.X, self.y, self.groups))
        self.assertEqual(len(splits), n_splits * n_repeats)

    def test_reproducibility(self):
        rgkf1 = RepeatedGroupKFold(n_splits=2, n_repeats=2, random_state=123)
        rgkf2 = RepeatedGroupKFold(n_splits=2, n_repeats=2, random_state=123)

        splits1 = list(rgkf1.split(self.X, self.y, self.groups))
        splits2 = list(rgkf2.split(self.X, self.y, self.groups))

        for (tr1, te1), (tr2, te2) in zip(splits1, splits2):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)


class TestExtendedGroupShuffleSplit(unittest.TestCase):
    """Tests for `ExtendedGroupShuffleSplit` cross-validator and its edge cases."""

    def setUp(self):
        self.groups = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
        self.X = pd.DataFrame({
            "feat_group": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            "feature_2": np.random.randn(16)
        })
        self.y = np.array([10, 10, 2, 2, 5, 5, 1, 1, 8, 8, 3, 3, 4, 4, 9, 9])

    def test_get_n_splits(self):
        egss = ExtendedGroupShuffleSplit(n_splits=7)
        self.assertEqual(egss.get_n_splits(), 7)

    def test_basic_splitting_without_replacement(self):
        egss = ExtendedGroupShuffleSplit(n_splits=3, n_test_groups=1, replace=False, random_state=42)
        splits = list(egss.split(self.X, self.y, self.groups))
        self.assertEqual(len(splits), 3)

        tested_groups_per_split = []
        for train_idx, test_idx in splits:
            test_grps = set(self.groups[test_idx])
            self.assertEqual(len(test_grps), 1)
            tested_groups_per_split.append(tuple(sorted(test_grps)))

        # Unique combinations tested because replace=False
        self.assertEqual(len(set(tested_groups_per_split)), 3)

    def test_float_n_test_groups(self):
        # 4 unique groups * 0.5 = 2 test groups
        egss = ExtendedGroupShuffleSplit(n_splits=1, n_test_groups=0.5, random_state=42)
        for _, test_idx in egss.split(self.X, self.y, self.groups):
            self.assertEqual(len(np.unique(self.groups[test_idx])), 2)

    def test_train_groups_fixed(self):
        egss = ExtendedGroupShuffleSplit(
            n_splits=2, n_test_groups=1, train_groups=[1], random_state=42
        )
        for train_idx, test_idx in egss.split(self.X, self.y, self.groups):
            train_grps = self.groups[train_idx]
            test_grps = self.groups[test_idx]
            
            # Group 1 must be present in training
            self.assertIn(1, train_grps)
            # Group 1 must never leak into test
            self.assertNotIn(1, test_grps)

    def test_max_test_samples_per_group(self):
        egss = ExtendedGroupShuffleSplit(
            n_splits=1,
            n_test_groups=2,
            max_test_samples_per_group=2,
            random_state=42
        )
        for _, test_idx in egss.split(self.X, self.y, self.groups):
            # 2 test groups, maximum 2 samples per group = 4 total test samples
            self.assertEqual(len(test_idx), 4)

    def test_reinjection_random_strategy(self):
        egss = ExtendedGroupShuffleSplit(
            n_splits=1,
            n_test_groups=1,
            train_groups=[1],
            max_test_samples_per_group=1,
            n_inject_samples=2,
            injection_strategy="random",
            random_state=42
        )
        for train_idx, test_idx in egss.split(self.X, self.y, self.groups):
            # Test has 1 sample
            self.assertEqual(len(test_idx), 1)
            # Train has: 4 base samples (group 1) + 2 injected samples = 6 samples
            self.assertEqual(len(train_idx), 6)

    def test_reinjection_lowest_target_strategy_pandas(self):
        egss = ExtendedGroupShuffleSplit(
            n_splits=1,
            n_test_groups=1,
            train_groups=[1],
            max_test_samples_per_group=1,
            n_inject_samples=1,
            injection_strategy="lowest_target",
            feature_group_col="feat_group",
            random_state=42
        )
        for train_idx, test_idx in egss.split(self.X, self.y, self.groups):
            self.assertEqual(len(test_idx), 1)
            self.assertEqual(len(train_idx), 5)  # 4 base + 1 injected

    def test_reinjection_lowest_target_strategy_numpy(self):
        X_np = self.X.to_numpy()
        egss = ExtendedGroupShuffleSplit(
            n_splits=1,
            n_test_groups=1,
            train_groups=[1],
            max_test_samples_per_group=1,
            n_inject_samples=1,
            injection_strategy="lowest_target",
            feature_group_col=0,
            random_state=42
        )
        for train_idx, test_idx in egss.split(X_np, self.y, self.groups):
            self.assertEqual(len(test_idx), 1)
            self.assertEqual(len(train_idx), 5)

    def test_sample_and_reject_branch(self):
        # Force low threshold to trigger sample-and-reject logic branch
        egss = ExtendedGroupShuffleSplit(
            n_splits=3,
            n_test_groups=1,
            replace=False,
            max_combinations_exhaustive=2,
            random_state=42
        )
        splits = list(egss.split(self.X, self.y, self.groups))
        self.assertEqual(len(splits), 3)

    # --- Edge Cases & Validation Errors ---

    def test_invalid_n_test_groups_float(self):
        egss = ExtendedGroupShuffleSplit(n_test_groups=1.5)
        with self.assertRaises(ValueError):
            list(egss.split(self.X, self.y, self.groups))

    def test_invalid_n_test_groups_type(self):
        egss = ExtendedGroupShuffleSplit(n_test_groups="invalid")
        with self.assertRaises(TypeError):
            list(egss.split(self.X, self.y, self.groups))

    def test_no_candidate_groups_raises_error(self):
        # Setting all groups as training groups leaves no testing candidate groups
        egss = ExtendedGroupShuffleSplit(train_groups=[1, 2, 3, 4])
        with self.assertRaises(ValueError) as ctx:
            list(egss.split(self.X, self.y, self.groups))
        self.assertIn("No candidate groups remaining", str(ctx.exception))

    def test_n_test_groups_out_of_bounds(self):
        egss = ExtendedGroupShuffleSplit(n_test_groups=10)
        with self.assertRaises(ValueError):
            list(egss.split(self.X, self.y, self.groups))

    def test_invalid_max_test_samples_per_group(self):
        egss = ExtendedGroupShuffleSplit(max_test_samples_per_group=0)
        with self.assertRaises(ValueError):
            list(egss.split(self.X, self.y, self.groups))

    def test_missing_y_for_lowest_target_strategy(self):
        egss = ExtendedGroupShuffleSplit(
            n_test_groups=1,
            max_test_samples_per_group=1,
            n_inject_samples=1,
            injection_strategy="lowest_target",
            feature_group_col="feat_group"
        )
        with self.assertRaises(ValueError) as ctx:
            list(egss.split(self.X, y=None, groups=self.groups))
        self.assertIn("Target 'y' must be provided", str(ctx.exception))

    def test_missing_feature_group_col_for_lowest_target_strategy(self):
        egss = ExtendedGroupShuffleSplit(
            n_test_groups=1,
            max_test_samples_per_group=1,
            n_inject_samples=1,
            injection_strategy="lowest_target",
            feature_group_col=None
        )
        with self.assertRaises(ValueError) as ctx:
            list(egss.split(self.X, y=self.y, groups=self.groups))
        self.assertIn("feature_group_col' must be provided", str(ctx.exception))

    def test_unknown_injection_strategy(self):
        egss = ExtendedGroupShuffleSplit(
            n_test_groups=1,
            max_test_samples_per_group=1,
            n_inject_samples=1,
            injection_strategy="unsupported_strategy"
        )
        with self.assertRaises(ValueError) as ctx:
            list(egss.split(self.X, y=self.y, groups=self.groups))
        self.assertIn("Unknown injection strategy", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()