# -*- coding: utf-8 -*-
"""
Unit tests for TargetSortedGroupShuffleSplit cross-validation splitter.
"""

import sys
import unittest
import numpy as np
import pandas as pd

sys.path.append(r"../scr/")

from custom_splitter import TargetSortedGroupShuffleSplit


class TestTargetSortedGroupShuffleSplit(unittest.TestCase):

    def setUp(self):
        """Set up standard test datasets (pandas DataFrame and numpy ndarray)."""
        # Dataset where stimulus average target values are known:
        # Group 0 (Train): indices 0..3 -> 'S1' (mean y = 10), 'S2' (mean y = 100)
        # Group 1 (Test):  indices 4..7 -> 'S1' (y = 5),  'S2' (y = 50)
        self.df_X = pd.DataFrame({
            "feature": [1, 2, 3, 4, 5, 6, 7, 8],
            "stim_id": ["S1", "S2", "S1", "S2", "S1", "S2", "S1", "S2"]
        })
        self.y = np.array([10, 100, 10, 100, 5, 50, 5, 50])
        self.groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        # 2D Numpy version of X (stim_id at column index 1)
        self.arr_X = np.array([
            [1, "S1"], [2, "S2"], [3, "S1"], [4, "S2"],
            [5, "S1"], [6, "S2"], [7, "S1"], [8, "S2"]
        ], dtype=object)

    def test_y_none_raises_value_error(self):
        """Ensure ValueError is raised if y is None."""
        splitter = TargetSortedGroupShuffleSplit(n_splits=1, test_size=1)
        with self.assertRaises(ValueError) as ctx:
            next(splitter.split(self.df_X, y=None, groups=self.groups))
        self.assertIn("Target variable 'y' must be provided", str(ctx.exception))

    def test_string_col_on_numpy_array_raises_error(self):
        """Ensure string column index on a numpy array raises ValueError."""
        splitter = TargetSortedGroupShuffleSplit(
            n_splits=1, test_size=1, stim_id_col="stim_id"
        )
        with self.assertRaises(ValueError) as ctx:
            next(splitter.split(self.arr_X, y=self.y, groups=self.groups))
        self.assertIn("was provided as a string", str(ctx.exception))

    def test_target_sorting_and_injection_logic(self):
        """Verify that samples with lowest average target value per stim_id are sent back."""
        # 2 unique groups (0 and 1). Group 0 = train, Group 1 = test.
        # Group 1 contains indices [4, 5, 6, 7] -> stims ['S1', 'S2', 'S1', 'S2'].
        # n_test_samples=2 keeps 2 test samples, leaving 2 unused test samples.
        # S1 has train mean y = 10, S2 has train mean y = 100.
        # Leftover sample with S1 (lower mean target) must be injected back first.
        splitter = TargetSortedGroupShuffleSplit(
            n_splits=1,
            test_size=1,
            train_size=1,
            n_test_samples=2,
            n_train_samples_back=1,
            stim_id_col="stim_id",
            random_state=42
        )

        train_idx, test_idx = next(splitter.split(self.df_X, self.y, self.groups))

        # Check total number of samples
        self.assertEqual(len(test_idx), 2)
        self.assertEqual(len(train_idx), 4 + 1)  # 4 original + 1 put back

        # Identify which test indices were left over
        all_test_group_indices = np.where(self.groups == 1)[0]
        leftovers = np.setdiff1d(all_test_group_indices, test_idx)

        # The put-back sample must be in train_idx and come from leftovers
        injected_sample = np.intersect1d(train_idx, leftovers)
        self.assertEqual(len(injected_sample), 1)

        # Injected sample's stim_id should be 'S1' because mean(S1)=10 < mean(S2)=100
        injected_stim = self.df_X.loc[injected_sample[0], "stim_id"]
        self.assertEqual(injected_stim, "S1")

    def test_numpy_array_input(self):
        """Ensure splitter works correctly with 2D numpy array and integer column index."""
        splitter = TargetSortedGroupShuffleSplit(
            n_splits=1,
            test_size=1,
            n_test_samples=2,
            n_train_samples_back=1,
            stim_id_col=1,
            random_state=42
        )
        train_idx, test_idx = next(splitter.split(self.arr_X, self.y, self.groups))
        
        self.assertEqual(len(test_idx), 2)
        self.assertEqual(len(train_idx), 5)

    def test_unseen_stim_id_ranks_last(self):
        """Ensure test samples with a stim_id not present in train set rank last (inf)."""
        df_unseen = pd.DataFrame({
            "feature": range(6),
            "stim_id": ["S1", "S1", "S1", "S1", "S_NEW", "S1"]
        })
        y = np.array([10, 10, 10, 10, 0, 10])  # S_NEW raw y=0, but S_NEW is unseen in train
        groups = np.array([0, 0, 0, 0, 1, 1])

        splitter = TargetSortedGroupShuffleSplit(
            n_splits=1,
            test_size=1,
            train_size=1,
            n_test_samples=1,
            n_train_samples_back=1,
            stim_id_col="stim_id",
            random_state=0
        )

        train_idx, test_idx = next(splitter.split(df_unseen, y, groups))

        leftovers_check = np.setdiff1d(np.where(groups == 1)[0], test_idx)
        if 4 in leftovers_check:
            self.assertNotIn(4, train_idx)

    def test_n_test_samples_all_yields_no_injection(self):
        """When n_test_samples='all', no leftover samples exist to put back."""
        splitter = TargetSortedGroupShuffleSplit(
            n_splits=1,
            test_size=1,
            n_test_samples="all",
            n_train_samples_back=2,
            stim_id_col="stim_id",
            random_state=42
        )
        train_idx, test_idx = next(splitter.split(self.df_X, self.y, self.groups))

        self.assertEqual(len(train_idx), 4)
        self.assertEqual(len(test_idx), 4)

    def test_n_train_samples_back_zero(self):
        """When n_train_samples_back=0, no leftovers are injected into training set."""
        splitter = TargetSortedGroupShuffleSplit(
            n_splits=1,
            test_size=1,
            n_test_samples=2,
            n_train_samples_back=0,
            stim_id_col="stim_id",
            random_state=42
        )
        train_idx, test_idx = next(splitter.split(self.df_X, self.y, self.groups))

        self.assertEqual(len(train_idx), 4)
        self.assertEqual(len(test_idx), 2)

    def test_indices_disjoint_and_sorted(self):
        """Ensure final train and test index arrays are sorted and disjoint."""
        splitter = TargetSortedGroupShuffleSplit(
            n_splits=3,
            test_size=1,
            n_test_samples=2,
            n_train_samples_back=1,
            stim_id_col="stim_id",
            random_state=42
        )

        for train_idx, test_idx in splitter.split(self.df_X, self.y, self.groups):
            # Check sorted order
            self.assertTrue(np.array_equal(train_idx, np.sort(train_idx)))
            self.assertTrue(np.array_equal(test_idx, np.sort(test_idx)))

            # Check no overlap
            overlap = np.intersect1d(train_idx, test_idx)
            self.assertEqual(len(overlap), 0)


if __name__ == "__main__":
    unittest.main()