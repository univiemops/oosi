# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:31:42 2026

@author: david
"""
import sys

sys.path.append(r"../scr/")

print("Test case 1: ")

import numpy as np
from custom_splitter import GroupKFold

X = np.array([[10], [20], [30], [40], [50], [60], [70], [80], [90], [100], [110], [120], [130], [140], [150], [160], [170]])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

group_kfold = GroupKFold(n_splits=2, shuffle=True, random_state=3141592)
for i, (train_index, test_index) in enumerate(group_kfold.split(X, y, groups)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups[train_index]}")
    print(f"  Test:  index={test_index}, group={groups[test_index]}")


print("\nTest case 2: ")

import numpy as np
from custom_splitter import RepeatedGroupKFold

X = np.array([[10], [20], [30], [40], [50], [60], [70], [80], [90], [100], [110], [120], [130], [140], [150], [160], [170]])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

rkf = RepeatedGroupKFold(n_splits=2, n_repeats=2, random_state=3141592)
for i, (train_index, test_index) in enumerate(rkf.split(X, y, groups)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups[train_index]}")
    print(f"  Test:  index={test_index}, group={groups[test_index]}")


print("\nTest case 3: ")

import numpy as np
from custom_splitter import RepeatedGroupKFold

X = np.array([[10], [20], [30], [40], [50], [60], [70], [80], [90], [100], [110], [120], [130], [140], [150], [160], [170]])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

rkf = RepeatedGroupKFold(n_splits=3, n_repeats=3, random_state=3141592)
for i, (train_index, test_index) in enumerate(rkf.split(X, y, groups)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups[train_index]}")
    print(f"  Test:  index={test_index}, group={groups[test_index]}")


print("\nTest case 4: ")

import numpy as np
from custom_splitter import CustomGroupShuffleSplit

X = np.array([[10], [20], [30], [40], [50], [60], [70], [80], [90], [100], [110], [120], [130], [140], [150], [160], [170]])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

gss_all = CustomGroupShuffleSplit(n_splits=5, test_size=1, n_test_samples="all", random_state=3141592)
for i, (train_index, test_index) in enumerate(gss_all.split(X, y, groups)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups[train_index]}")
    print(f"  Test:  index={test_index}, group={groups[test_index]}")

gss_all = CustomGroupShuffleSplit(n_splits=3, test_size=2, n_test_samples="all", random_state=3141592)
for i, (train_index, test_index) in enumerate(gss_all.split(X, y, groups)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups[train_index]}")
    print(f"  Test:  index={test_index}, group={groups[test_index]}")

gss_sub = CustomGroupShuffleSplit(n_splits=3, test_size=2, n_test_samples=2, n_train_samples_back=0, random_state=3141592)
for i, (train_index, test_index) in enumerate(gss_sub.split(X, y, groups)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups[train_index]}")
    print(f"  Test:  index={test_index}, group={groups[test_index]}")

gss_inject = CustomGroupShuffleSplit(n_splits=2, test_size=2, n_test_samples=2, n_train_samples_back=2, random_state=3141592)
for i, (train_index, test_index) in enumerate(gss_inject.split(X, y, groups)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}, group={groups[train_index]}")
    print(f"  Test:  index={test_index}, group={groups[test_index]}")
