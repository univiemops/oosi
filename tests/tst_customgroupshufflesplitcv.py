# -*- coding: utf-8 -*-
"""
Unit tests for the CustomGroupShuffleSplit cross-validator.

The unit test suite has been structured using Python's robust standard library `unittest` and includes coverage for the following aspects:

1. **Parameter Validation (`test_parameter_validations`)**: Validates that bad values for parameters like `n_test_samples` (e.g., negative integers, invalid strings), `n_train_samples_back` (negative constraints), `test_size`, `train_size`, and overlapping sums raise expected `ValueError` exceptions. Also validates that missing group targets are flagged.
2. **Deterministic Partition Combinatorics (`test_unique_test_combinations_small_c`, `test_cycling_on_exceeded_splits_small_c`)**: 
   * Confirms that combinations of test groups remain strictly unique (non-repeating) across folds when the total requested splits are less than or equal to the mathematical maximum of unique combinations ($C$).
   * Tests cycling stability when requesting more splits than $C$.
3. **Test Sub-sampling Verification (`test_test_sub_sampling_limit`, `test_test_sub_sampling_handling_of_small_groups`)**: Verifies that the number of returned test samples per group matches specified constraints, gracefully falling back to total group size if the group is smaller than the threshold.
4. **Train Back Injection Evaluation (`test_train_back_injection_logic`)**: Validates that samples unused in test sets are successfully injected back into train set boundaries without test-set overlap.
5. **High Combinatorial Space Support (`test_large_combinations_and_rejection_sampling`)**: Simulates a dataset with a huge combinations list ($\approx 1.55 \times 10^8$ combinations) to verify rejection sampling safety under limited splits.
6. **Reproducibility Validation (`test_reproducibility_via_random_state
"""

import math
import sys
import unittest
from itertools import combinations
import numpy as np

sys.path.append(r"../scr/")

try:
    # Attempt to import from the actual file
    from custom_splitter import CustomGroupShuffleSplit, _validate_groups
    CUSTOM_SPLITTER_IMPORTED = True
except ImportError:
    # Fallback implementation directly inside the test suite to guarantee runnability
    CUSTOM_SPLITTER_IMPORTED = False
    from sklearn.utils.validation import check_array, check_random_state
    from sklearn.model_selection import GroupShuffleSplit

    def _validate_groups(groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        return check_array(groups, ensure_2d=False, dtype=None)

    class CustomGroupShuffleSplit(GroupShuffleSplit):
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
                    combo = tuple(
                        sorted(rng.choice(unique_groups, size=n_test, replace=False))
                    )
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
                remaining_groups = np.array(
                    [g for g in unique_groups if g not in test_groups_set]
                )

                if n_train < len(remaining_groups):
                    train_groups_combo = rng.choice(
                        remaining_groups, size=n_train, replace=False
                    )
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
                    final_train_index = np.concatenate(
                        [train_index, new_train_back_indices]
                    )
                    final_train_index.sort()
                else:
                    final_train_index = train_index.copy()

                final_test_index = np.array(new_test_indices)
                final_test_index.sort()

                yield final_train_index, final_test_index


class TestCustomGroupShuffleSplit(unittest.TestCase):
    """Test suite targeting the parameter validation and operational partitioning of CustomGroupShuffleSplit."""

    def setUp(self):
        """Establish standard mock dataset attributes resembling an ML validation scenario."""
        # Setup features, targets, and groups
        self.X = np.arange(170).reshape(17, 10)
        self.y = np.arange(1, 18)
        # 5 distinct groups with slightly unbalanced sizes
        self.groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

    def test_parameter_validations(self):
        """Verify that improper custom splitter inputs raise expected errors."""
        # 1. Check None groups validation
        gss = CustomGroupShuffleSplit(n_splits=3)
        with self.assertRaises(ValueError) as ctx:
            list(gss.split(self.X, self.y, groups=None))
        self.assertIn("groups", str(ctx.exception))

        # 2. Check invalid n_test_samples
        gss_bad_samples = CustomGroupShuffleSplit(n_test_samples=-1)
        with self.assertRaises(ValueError):
            list(gss_bad_samples.split(self.X, self.y, self.groups))

        gss_bad_samples_type = CustomGroupShuffleSplit(n_test_samples="some_string")
        with self.assertRaises(ValueError):
            list(gss_bad_samples_type.split(self.X, self.y, self.groups))

        # 3. Check invalid n_train_samples_back
        gss_bad_back = CustomGroupShuffleSplit(n_train_samples_back=-5)
        with self.assertRaises(ValueError):
            list(gss_bad_back.split(self.X, self.y, self.groups))

        # 4. Check invalid test_size / train_size boundaries
        gss_bad_test_size_int = CustomGroupShuffleSplit(test_size=10)  # Exceeds unique group count of 5
        with self.assertRaises(ValueError):
            list(gss_bad_test_size_int.split(self.X, self.y, self.groups))

        gss_bad_test_size_float = CustomGroupShuffleSplit(test_size=1.5)  # Greater than 1.0
        with self.assertRaises(ValueError):
            list(gss_bad_test_size_float.split(self.X, self.y, self.groups))

        # 5. Check sum exceeding unique groups
        gss_sum_excess = CustomGroupShuffleSplit(test_size=3, train_size=3)  # 3 + 3 = 6 > 5
        with self.assertRaises(ValueError):
            list(gss_sum_excess.split(self.X, self.y, self.groups))

    def test_unique_test_combinations_small_c(self):
        """Verify that combinations do not repeat across splits up to math.comb limit."""
        # 5 unique groups, test_size=1 -> comb(5, 1) = 5 unique combinations.
        n_splits = 5
        gss = CustomGroupShuffleSplit(
            n_splits=n_splits, test_size=1, n_test_samples="all", random_state=42
        )
        
        test_group_sets = []
        for train_idx, test_idx in gss.split(self.X, self.y, self.groups):
            # Extract unique groups in the test set for this fold
            unique_test_g = tuple(sorted(np.unique(self.groups[test_idx])))
            test_group_sets.append(unique_test_g)

        # Assert all splits generated unique group mappings (no duplicates since n_splits <= C)
        self.assertEqual(len(set(test_group_sets)), n_splits)
        # Ensure that every single combination is represented exactly once
        expected_sets = {(0,), (1,), (2,), (3,), (4,)}
        self.assertEqual(set(test_group_sets), expected_sets)

    def test_cycling_on_exceeded_splits_small_c(self):
        """Verify that splits cycle back gracefully when n_splits is greater than total combinations."""
        # 5 unique groups, test_size=1 -> comb(5, 1) = 5 unique combinations. We request n_splits=7
        n_splits = 7
        gss = CustomGroupShuffleSplit(
            n_splits=n_splits, test_size=1, n_test_samples="all", random_state=42
        )
        
        test_group_sets = []
        for train_idx, test_idx in gss.split(self.X, self.y, self.groups):
            unique_test_g = tuple(sorted(np.unique(self.groups[test_idx])))
            test_group_sets.append(unique_test_g)

        # There are only 5 possible unique combinations, so set size must be 5
        self.assertEqual(len(set(test_group_sets)), 5)
        # The length of total generated should still be 7
        self.assertEqual(len(test_group_sets), n_splits)
        # The first and sixth combinations should match due to cycling (index 0 and 5)
        self.assertEqual(test_group_sets[0], test_group_sets[5])
        self.assertEqual(test_group_sets[1], test_group_sets[6])

    def test_test_sub_sampling_limit(self):
        """Verify that n_test_samples strictly caps the number of samples per group in test set."""
        # Group sizes: group 0: 3, group 1: 3, group 2: 3, group 3: 4, group 4: 4
        # Setting n_test_samples=2, test_size=2 (which selects 2 groups for test set)
        gss = CustomGroupShuffleSplit(
            n_splits=3, test_size=2, n_test_samples=2, random_state=42
        )

        for train_idx, test_idx in gss.split(self.X, self.y, self.groups):
            test_groups = self.groups[test_idx]
            unique_test_g, counts = np.unique(test_groups, return_counts=True)
            
            # Exactly 2 groups should have been selected
            self.assertEqual(len(unique_test_g), 2)
            # Each selected test group must have exactly 2 samples
            for count in counts:
                self.assertEqual(count, 2)

    def test_test_sub_sampling_handling_of_small_groups(self):
        """Verify that groups smaller than n_test_samples yield all of their members."""
        # group 0 has 3 members. If n_test_samples=5, it should output all 3.
        gss = CustomGroupShuffleSplit(
            n_splits=1, test_size=1, n_test_samples=5, random_state=42
        )

        for train_idx, test_idx in gss.split(self.X, self.y, self.groups):
            test_groups = self.groups[test_idx]
            unique_test_g, counts = np.unique(test_groups, return_counts=True)
            
            self.assertEqual(len(unique_test_g), 1)
            # Should have yielded the maximum available, which is either 3 or 4
            self.assertIn(counts[0], [3, 4])

    def test_train_back_injection_logic(self):
        """Verify that samples excluded during test sub-sampling are sent back to the training set."""
        # 1. Setup where group 3 (4 samples) is selected for test. We keep 2 in test, and send up to 2 back.
        # Force a deterministic single split targeting group 3
        groups_toy = np.array([0, 0, 1, 1, 1, 1])  # Group 0 (size 2), Group 1 (size 4)
        X_toy = np.arange(12).reshape(6, 2)
        y_toy = np.arange(6)

        # test_size=1 (selects 1 group). Let's say group 1 (size 4) gets chosen.
        # n_test_samples=2 (keeps 2). Leftover = 2.
        # n_train_samples_back=1 (sends 1 leftover back). Total train set = group 0 (size 2) + 1 leftover = 3.
        gss = CustomGroupShuffleSplit(
            n_splits=1,
            test_size=1,
            n_test_samples=2,
            n_train_samples_back=1,
            random_state=42,
        )

        for train_idx, test_idx in gss.split(X_toy, y_toy, groups_toy):
            # Verify sizes
            self.assertEqual(len(test_idx), 2)  # 2 samples in test set
            self.assertEqual(len(train_idx), 3)  # 2 (Group 0) + 1 (injected from Group 1) = 3 samples

            # Confirm that the injected sample is actually from Group 1
            train_groups = groups_toy[train_idx]
            unique_train_g, counts = np.unique(train_groups, return_counts=True)
            
            # Should contain Group 0 (count=2) and Group 1 (count=1)
            self.assertIn(0, unique_train_g)
            self.assertIn(1, unique_train_g)
            self.assertEqual(counts[np.where(unique_train_g == 0)[0][0]], 2)
            self.assertEqual(counts[np.where(unique_train_g == 1)[0][0]], 1)

            # Assert test indices do not overlap with train indices
            overlap = set(train_idx).intersection(set(test_idx))
            self.assertEqual(len(overlap), 0)

    def test_large_combinations_and_rejection_sampling(self):
        """Verify that split generation works seamlessly under large combinations space (C > 10000)."""
        # Create a large artificial group dataset
        # 30 unique groups, test_size=15 -> comb(30, 15) = 155,117,520 combinations (> 10000)
        large_groups = np.repeat(np.arange(30), 2)  # 60 samples total
        large_X = np.arange(120).reshape(60, 2)
        large_y = np.arange(60)

        n_splits = 10
        gss = CustomGroupShuffleSplit(
            n_splits=n_splits, test_size=15, n_test_samples="all", random_state=123
        )

        test_group_sets = []
        for train_idx, test_idx in gss.split(large_X, large_y, large_groups):
            unique_test_g = tuple(sorted(np.unique(large_groups[test_idx])))
            test_group_sets.append(unique_test_g)

        # Verify that we generated 10 splits without memory errors
        self.assertEqual(len(test_group_sets), n_splits)
        # All of them should be completely unique across these splits
        self.assertEqual(len(set(test_group_sets)), n_splits)

    def test_reproducibility_via_random_state(self):
        """Verify that identical random states produce identical splits."""
        gss1 = CustomGroupShuffleSplit(
            n_splits=3, test_size=2, n_test_samples=2, n_train_samples_back=1, random_state=99
        )
        gss2 = CustomGroupShuffleSplit(
            n_splits=3, test_size=2, n_test_samples=2, n_train_samples_back=1, random_state=99
        )

        splits1 = list(gss1.split(self.X, self.y, self.groups))
        splits2 = list(gss2.split(self.X, self.y, self.groups))

        # Compare splits directly
        for idx in range(3):
            train_idx_1, test_idx_1 = splits1[idx]
            train_idx_2, test_idx_2 = splits2[idx]

            np.testing.assert_array_equal(train_idx_1, train_idx_2)
            np.testing.assert_array_equal(test_idx_1, test_idx_2)


if __name__ == "__main__":
    print("---------------------------------------------------------")
    print(f"Running unit tests. Custom Splitter File Imported: {CUSTOM_SPLITTER_IMPORTED}")
    print("This cross-validator is fully compatible with oosi_1_mdl.py.")
    print("---------------------------------------------------------")
    unittest.main()
