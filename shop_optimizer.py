import json

from dataclasses import dataclass
from enum import Enum
from functools import cache
from itertools import combinations
from typing import List, Set

import numpy as np
import pandas as pd

from aenum import IntFlag, auto
from tqdm import tqdm


"""
ShopOptimizer loads all available shop data and hero data with the goal of finding the following information:

What is the maximum amount of silver that we can earn in the shop?
As a secondary criteria, can we also maximize the amount of gold that we can earn?
-- We'll save the secondary criteria as an exercise for later.
"""

class BuffEnum(Enum):
    dairy = 0
    decoration = 1
    drink = 2
    flower = 3
    food = 4
    gourmet = 5
    light = 6
    magic = 7
    necklace = 8
    shell = 9

"""
Our objective function is to maximize silver income, with a secondary objective of maximizing gold income.
Therefore, our comparison operators will be built around these assumptions.
"""
@dataclass
class Income:
    silver: float
    gold: float
    powder: float

    def __add__(self, other):
        if not isinstance(other, Income):
            raise TypeError(f"Cannot add Income and {type(other)}")
        return Income(silver=self.silver+other.silver, gold=self.gold+other.gold, powder=self.powder+other.powder)
    
    # radd operator must be implemented for sum() function to work properly.
    # https://stackoverflow.com/questions/1218710/pythons-sum-and-non-integer-values
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __mul__(self, other):
        # match other:
        match other:
            case int():
                return Income(silver=self.silver*other, gold=self.gold*other, powder=self.powder*other)
            case float():
                return Income(silver=self.silver*other, gold=self.gold*other, powder=self.powder*other)
            case Income():
                return Income(silver=self.silver*other.silver, gold=self.gold*other.gold, powder=self.powder*other.powder)
            case default:
                raise TypeError(f"Cannot multiply Income and {type(other)}")
    
    def __ge__(self, other):
        if not isinstance(other, Income):
            raise TypeError(f"Cannot compare Income and {type(other)}")
        return self.silver > other.silver or self.silver == other.silver and self.gold >= other.gold
    
    def __gt__(self, other):
        if not isinstance(other, Income):
            raise TypeError(f"Cannot compare Income and {type(other)}")
        return self.silver > other.silver or self.silver == other.silver and self.gold > other.gold or self.silver == other.silver and self.gold == other.gold and self.powder > other.powder

    def __lt__(self, other):
        return not self.__ge__(other)
    
    def __leq__(self, other):
        return not self.__gt__(other)
    
    def __eq__(self, other):
        if not isinstance(other, Income):
            raise TypeError(f"Cannot compare Income and {type(other)}")
        return self.silver == other.silver and self.gold == other.gold and self.powder == other.powder

@dataclass
class HeroV2:
    name: str
    buffs: dict[BuffEnum]


"""
Version 2 of the shop optimizer.
No longer tries to find all combinations of heroes and items.
Instead, we generate all possible combination of heroes, and fix the items as a weighted matrix of size: (num_items, num_buffs)
"""
class ShopOptimizerV2:
    def __init__(self):
        self.__load_data()
        self.__build_matrices()

    def __load_data(self):
        with open('shop_values.json') as f:
            shop_data = json.load(f)
        
        self.shop_dict = {
            data['name']: ShopItem(
                name=data['name'],
                income=Income(
                    silver=data['values']['silver'],
                    gold=data['values']['gold'],
                    powder=data['values']['powder']
                ),
                buff_types={BuffEnum[buff] for buff in data['types']}
            )
            for data in shop_data
        }

        self.shop_item_names = list(self.shop_dict.keys())

        with open('hero_shop_modifiers.json') as f:
            hero_data = json.load(f)

        self.hero_dict = {
            data['name']: HeroV2(
                name = data['name'],
                buffs = {
                    BuffEnum[buff_type]: value
                    for buff_type, value in data['buffs'].items()
                }
            )
            for data in hero_data
        }

        self.hero_names = list(self.hero_dict.keys())
    
    
    #                                magic     shell        gourmet                  
    # hero_combined(1,2,3,16,17)      0.42        0.3           0.2   
    # -> base   magic shell gourmet
    #    1       .42      .3    .2  .....  ->
    def __build_matrices(self):
        # Build the item matrix
        self.item_matrix = np.array([[
                shop_item.income.silver 
                if buff in shop_item.buff_types
                else 0 
                for buff in BuffEnum
            ]
            for shop_item in self.shop_dict.values()
        ])
        self.item_matrix = np.hstack((
            np.array([shop_item.income.silver for shop_item in self.shop_dict.values()]).reshape(-1,1),
            self.item_matrix
        ))

        self.hero_matrix = np.array([[
                hero.buffs[buff]
                if buff in hero.buffs.keys()
                else 0 
                for buff in BuffEnum
            ]
            for hero in self.hero_dict.values()
        ])

        # For our numpy array slicing to work correctly, we need to spawn index masks like so
        self.hero_combos = [list(mask) for mask in combinations(range(len(self.hero_names)), r=5)]


    def run(self):
        max_profit, item_idxs, hero_idxs = 0, [], []
        for hero_combo in tqdm(self.hero_combos):
            profit, item_combo = self.__eval_hero_item_set(hero_combo)
            if profit > max_profit:
                max_profit = profit
                item_idxs, hero_idxs = item_combo, hero_combo
                
        return max_profit, item_idxs, hero_idxs
    
    def __eval_hero_item_set(self, hero_combo):
        hero_submatrix = self.hero_matrix[hero_combo]
        hero_vector = np.sum(hero_submatrix, axis=0)
        # Prepend 1 to the vector to account for base income dot product
        hero_vector = np.insert(hero_vector, 0, 1, axis=0)
        prod = self.item_matrix @ hero_vector
        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        top_six_indices = np.argpartition(prod, -6)[-6:]
        profit = np.sum(prod[top_six_indices])
        return profit, list(top_six_indices)

opt = ShopOptimizerV2()
max_profit, item_idxs, hero_idxs = opt.run()
print(f"Max Profit: {max_profit}")
# print(f"Item indices: {item_idxs}")
# print(f"Hero indices: {hero_idxs}")

heroes = [opt.hero_names[idx] for idx in hero_idxs]
items = [opt.shop_item_names[idx] for idx in item_idxs]

print(f"Hero names: {heroes}")
print(f"Item names: {items}")
