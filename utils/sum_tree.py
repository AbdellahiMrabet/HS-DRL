# utils/sum_tree.py - SumTree for Prioritized Experience Replay

import numpy as np


class SumTree:
    """SumTree data structure for Prioritized Experience Replay (PER)"""
    
    __slots__ = ['capacity', 'tree', 'data', 'write', 'size']
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.size = 0
    
    def add(self, priority: float, data):
        """Add data with given priority"""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self._update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def _update(self, idx: int, priority: float):
        """Update tree with new priority"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def _propagate(self, idx: int, change: float):
        """Propagate change up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def get(self, s: float):
        """Get data by cumulative probability"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return self.data[data_idx]
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve index by cumulative probability"""
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(left + 1, s - self.tree[left])
    
    @property
    def total(self) -> float:
        """Total priority sum"""
        return self.tree[0]
