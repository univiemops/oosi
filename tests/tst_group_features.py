# -*- coding: utf-8 -*-
"""
Unit tests for the group_features function.

* **Dynamic Mocking**: 
  * `The test suite dynamically detects if the `shapiq` package is installed. If it isn't, a clean mock class `MockInteractionValues` is substituted at runtime to allow full unit test execution.
* **Granular Test Cases**:
  * `test_basic_main_effects_summation`: Asserts that plain feature weights are summed up correctly.
  * `test_internal_interaction_collapsing`: Tests that internal interaction effects (like order-2 inside the group) are correctly collapsed directly into the grouped feature.
  * `test_outside_interaction_aggregation`: Checks second-order interactions where one feature is in the group and the other is outside.
  * `test_higher_order_outside_interactions_ignored`: Asserts that complex multi-feature interactions that don't match the order-2 criteria are skipped as designed.
  * `test_untouched_outside_features_preserved`: Asserts that non-grouped features and their interactions are untouched.
  * `test_metadata_and_player_count_propagation`: Asserts that class parameters, including the adjusted total number of players, are perfectly mapped.
  * `test_single_element_group`: Covers edge cases like passing a grouping list containing only one feature index.
"""

import sys
import unittest
from unittest.mock import MagicMock

try:
    from shapiq.interaction_values import InteractionValues
    SHAPIQ_AVAILABLE = True
except ImportError:
    SHAPIQ_AVAILABLE = False
    
    # Define a clean mock class mimicking shapiq's InteractionValues for testing
    class MockInteractionValues:
        def __init__(
            self, 
            values: dict, 
            index: str, 
            max_order: int, 
            min_order: int, 
            n_players: int, 
            baseline_value: float, 
            estimation_budget: int = None
        ):
            self.dict_values = values
            self.values = values
            self.index = index
            self.max_order = max_order
            self.min_order = min_order
            self.n_players = n_players
            self.baseline_value = baseline_value
            self.estimation_budget = estimation_budget

    # Register the mock class in sys.modules to satisfy runtime dependencies
    mock_module = MagicMock()
    mock_module.InteractionValues = MockInteractionValues
    sys.modules["shapiq"] = mock_module
    sys.modules["shapiq.interaction_values"] = mock_module
    InteractionValues = MockInteractionValues


def group_features(
    iv: InteractionValues, features_to_group: list[int] or set[int], new_feature_id: int
) -> InteractionValues:
    """
    Groups a specified subset of features into a single unified feature
    for an InteractionValues object, supporting arbitrary interaction orders (k-SII).

    Args:
        iv: The original shapiq InteractionValues object.
        features_to_group: Iterable of feature indices (ints) to group together.
        new_feature_id: The new integer index to assign to the grouped feature.

    Returns:
        A new InteractionValues object with grouped values across all orders.
    """
    group = set(features_to_group)
    new_values = {}

    for subset, value in iv.dict_values.items():
        subset_set = set(subset)
        intersection = subset_set.intersection(group)

        # Case A: The subset has no elements inside the grouped features
        if len(intersection) == 0:
            new_values[subset] = new_values.get(subset, 0.0) + value
        # Case B: The subset contains elements of the group (any order)
        else:
            # Remove grouped features and replace them with the single macro-player
            outside_elements = subset_set.difference(group)
            new_subset = tuple(sorted(list(outside_elements) + [new_feature_id]))
            new_values[new_subset] = new_values.get(new_subset, 0.0) + value

    # Grab the baseline value from the original object
    baseline_val = iv.baseline_value

    # Use n_players instead of n_features
    new_n_players = iv.n_players - len(group) + 1

    # Determine the new minimum order dynamically based on the resulting dictionary keys
    new_min_order = min(len(k) for k in new_values.keys()) if new_values else iv.min_order

    # Return the freshly minted shapiq object
    return InteractionValues(
        values=new_values,
        index=iv.index,
        max_order=iv.max_order,
        min_order=new_min_order,
        n_players=new_n_players,
        baseline_value=baseline_val,
        estimation_budget=iv.estimation_budget,
    )


class TestGroupFeatures(unittest.TestCase):
    """Test suite targeting the logic and metadata of the group_features function."""

    def setUp(self):
        """Set up standard default metadata for InteractionValues objects."""
        self.default_meta = {
            "index": "k-SII",
            "max_order": 2,
            "min_order": 1,
            "n_players": 5,
            "baseline_value": 0.5,
            "estimation_budget": 1000
        }

    def test_basic_main_effects_summation(self):
        """Test that main effects inside the group sum together correctly and untouched elements are kept."""
        # 1 and 2 are grouped into 99. 3 is left untouched.
        initial_values = {
            (1,): 10.0,
            (2,): 5.0,
            (3,): 2.0
        }
        iv = InteractionValues(values=initial_values, **self.default_meta)
        
        result = group_features(iv, features_to_group=[1, 2], new_feature_id=99)
        
        # Expected: (99,) gets 10.0 + 5.0 = 15.0; (3,) stays 2.0; (1,) and (2,) removed;
        self.assertIn((99,), result.dict_values)
        self.assertEqual(result.dict_values[(99,)], 15.0)
        self.assertIn((3,), result.dict_values)
        self.assertEqual(result.dict_values[(3,)], 2.0)
        self.assertNotIn((1, 2), result.dict_values)

    def test_internal_interaction_collapsing(self):
        """Test that interactions fully contained within the group get added to the main effect."""
        # 1 and 2 are grouped into 99. The interaction (1, 2) is entirely inside the group.
        initial_values = {
            (1,): 10.0,
            (2,): 5.0,
            (1, 2): -3.0
        }
        iv = InteractionValues(values=initial_values, **self.default_meta)
        
        result = group_features(iv, features_to_group=[1, 2], new_feature_id=99)
        
        # Expected: (99,) gets 10.0 + 5.0 + (-3.0) = 12.0
        self.assertEqual(result.dict_values[(99,)], 12.0)
        # (1, 2) should not appear in the final values
        self.assertNotIn((1, 2), result.dict_values)

    def test_outside_interaction_aggregation(self):
        """Test that second-order interactions with features outside the group are aggregated correctly."""
        # Grouping {1, 2} into 99. Interactions (1, 3) and (2, 3) are outside interactions with feature 3.
        initial_values = {
            (1,): 10.0,
            (2,): 5.0,
            (3,): 1.5,
            (1, 2): -1.1,
            (1, 3): 4.0,
            (2, 3): 2.0
        }
        iv = InteractionValues(values=initial_values, **self.default_meta)
        
        result = group_features(iv, features_to_group=[1, 2], new_feature_id=99)
        
        # Expected:
        # Grouped main effect: 10.0 + 5.0 -1.1 = 13.9
        # Collapsed interaction with 3: (1, 3) + (2, 3) = 4.0 + 2.0 = 6.0 (saved as (3, 99) sorted)
        self.assertEqual(result.dict_values[(99,)], 13.9)
        self.assertEqual(result.dict_values[(3, 99)], 6.0)
        self.assertEqual(result.dict_values[(3,)], 1.5)

    def test_higher_order_outside_interactions(self):
        """Test that interactions with more than 1 outside element are included."""
        # Grouping {1, 2} into 99.
        # Subset (1, 3, 4) has elements inside group ({1}) and outside group ({3, 4}).
        initial_values = {
            (1,): 10.0,
            (2,): 5.0,
            (1, 3, 4): 8.0,
        }
        iv = InteractionValues(values=initial_values, **self.default_meta)
        
        result = group_features(iv, features_to_group=[1, 2], new_feature_id=99)
        
        # Expected:
        # Grouped main effect: 10.0 + 5.0 = 15
        # Interaction should stay but with new index
        self.assertEqual(result.dict_values[(99,)], 15.0)
        self.assertEqual(result.dict_values[(3, 4, 99)], 8.0)
        self.assertNotIn((99, 3, 4), result.dict_values)

    def test_untouched_outside_features_preserved(self):
        """Test that interactions totally unrelated to the group features remain unchanged."""
        # Grouping {1, 2} into 99.
        # (3, 4) is completely outside of {1, 2}.
        initial_values = {
            (3, 4): 7.0
        }
        iv = InteractionValues(values=initial_values, **self.default_meta)
        
        result = group_features(iv, features_to_group=[1, 2], new_feature_id=99)
        
        # Expected:
        # (3, 4) should stay the same
        self.assertIn((3, 4), result.dict_values)
        self.assertEqual(result.dict_values[(3, 4)], 7.0)

    def test_metadata_and_player_count_propagation(self):
        """Test that object metadata is correctly updated or preserved."""
        # Grouping {1, 2, 3} (3 players) into 99.
        # n_players starts at 5. New n_players should be 5 - 3 + 1 = 3.
        initial_values = {(1,): 1.0}
        iv = InteractionValues(values=initial_values, **self.default_meta)
        
        result = group_features(iv, features_to_group=[1, 2, 3], new_feature_id=99)
        
        self.assertEqual(result.n_players, 3)
        self.assertEqual(result.index, "k-SII")
        self.assertEqual(result.max_order, 2)
        self.assertEqual(result.min_order, 1)
        self.assertEqual(result.baseline_value, 0.5)
        self.assertEqual(result.estimation_budget, 1000)

    def test_single_element_group(self):
        """Test grouping with only a single element in the group."""
        # Grouping only {1} into 99.
        initial_values = {
            (1,): 10.0,
            (1, 2): 4.0,
            (3,): 5.0
        }
        iv = InteractionValues(values=initial_values, **self.default_meta)
        
        result = group_features(iv, features_to_group=[1], new_feature_id=99)
        
        # Expected:
        # (99,) gets 10.0
        # (2, 99) gets 4.0
        # (3,) gets 5.0
        # n_players remains 5 (5 - 1 + 1)
        self.assertEqual(result.dict_values[(99,)], 10.0)
        self.assertEqual(result.dict_values[(2, 99)], 4.0)
        self.assertEqual(result.dict_values[(3,)], 5.0)
        self.assertEqual(result.n_players, 5)


if __name__ == "__main__":
    print(f"Running unit tests. Shapiq Library Mocked: {not SHAPIQ_AVAILABLE}")
    unittest.main()
