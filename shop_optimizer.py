import heapq
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

class BuffFlag(IntFlag):
    dairy = auto()
    decoration = auto()
    drink = auto()
    flower = auto()
    food = auto()
    gourmet = auto()
    light = auto()
    magic = auto()
    necklace = auto()
    shell = auto()


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

@dataclass
class ShopItem:
    name: str
    income: Income
    buff_types: BuffFlag


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
        # Build the item matrix, factoring in all buffs
        self.item_prices = np.array([shop_item.income.silver for shop_item in self.shop_dict.values()])
        self.item_matrix = np.array([[
                shop_item.income.silver
                if buff in shop_item.buff_types
                else 0
                for buff in BuffEnum
            ]
            for shop_item in self.shop_dict.values()
        ])

        # # Prepend the base income to the item matrix
        # self.item_matrix = np.hstack((
        #     np.array([shop_item.income.silver for shop_item in self.shop_dict.values()]).reshape(-1,1),
        #     self.item_matrix
        # ))

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
        self.item_combos = [list(mask) for mask in combinations(range(len(self.shop_item_names)), r=6)]


    def run(self, hero_combo=True):
        if hero_combo:
            return self.__run_hero_combo()
        else:
            return self.__run_item_combo()


    def __run_hero_combo(self):
        max_profit, item_idxs, hero_idxs = 0, [], []

        # # Prepend the base income to the item matrix
        item_matrix = np.hstack((
            np.array([shop_item.income.silver for shop_item in self.shop_dict.values()]).reshape(-1,1),
            self.item_matrix
        ))
        # print(item_matrix.shape)

        self.profits = []
        for hero_combo in tqdm(self.hero_combos):
            profit, item_combo = self.__eval_hero_item_set(hero_combo, item_matrix)
            heapq.heappush(self.profits, (profit, item_combo, hero_combo))
            if profit >= max_profit:
                max_profit = profit
                item_idxs, hero_idxs = item_combo, hero_combo

        return max_profit, item_idxs, hero_idxs

    def __run_item_combo(self):
        max_profit, item_idxs, hero_idxs = 0, [], []
        # X = self.hero_matrix
        # hero_matrix = np.hstack((np.ones((X.shape[0], 1)), X))

        self.profits = []
        # We want to generate all combinations of 5 heroes out of 7.
        candidate_hero_indices = [list(mask) for mask in combinations(range(7), r=5)]

        for item_combo in tqdm(self.item_combos):
            for profit, hero_combo in self.__eval_item_hero_set(item_combo, candidate_hero_indices):
            # profit, hero_combo = self.__eval_item_hero_set(item_combo, candidate_hero_indices)
                heapq.heappush(self.profits, (profit, item_combo, hero_combo))
                if profit >= max_profit:
                    max_profit = profit
                    item_idxs, hero_idxs = item_combo, hero_combo

        return max_profit, item_idxs, hero_idxs


    # This variant takes a hero combo and evaluates against a fixed item matrix
    def __eval_hero_item_set(self, hero_combo, item_matrix):
        hero_submatrix = self.hero_matrix[hero_combo]
        hero_vector = np.sum(hero_submatrix, axis=0)
        # Prepend 1 to the vector to account for base income dot product
        hero_vector = np.insert(hero_vector, 0, 1, axis=0)
        prod = item_matrix @ hero_vector

        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        top_six_indices = np.argpartition(prod, -6)[-6:]
        profit = np.sum(prod[top_six_indices])
        return profit, list(top_six_indices)


    # This variant takes an item combo and evaluates against a fixed hero matrix
    def __eval_item_hero_set(self, item_combo, candidate_hero_indices):
        item_submatrix = self.item_matrix[item_combo]
        item_vector = np.sum(item_submatrix, axis=0)
        prod = self.hero_matrix @ item_vector

        base_profit = np.sum(self.item_prices[item_combo])
        top_seven_hero_indices = np.argpartition(prod, -7)[-7:]

        for hero_index_subset in candidate_hero_indices:
            five_hero_indices = top_seven_hero_indices[hero_index_subset]
            bonus_profit = np.sum(prod[five_hero_indices])
            profit = base_profit + bonus_profit
            yield profit, list(five_hero_indices)




def print_profit_info(opt, rank, max_profit, item_idxs, hero_idxs):
    heroes = sorted([opt.hero_names[idx] for idx in hero_idxs])
    items = sorted([opt.shop_item_names[idx] for idx in item_idxs])
    print(f"Results for rank {rank}. Profit: {max_profit}")
    print(f"Hero names: {heroes}")
    print(f"Item names: {items}")


opt = ShopOptimizerV2()
max_profit, item_idxs, hero_idxs = opt.run(hero_combo=False)


top_ten_profits = heapq.nlargest(10, opt.profits) #  key=lambda x: x[0]

for rank, (profit, item_idxs, hero_idxs) in enumerate(top_ten_profits):
    print_profit_info(opt, rank, profit, item_idxs, hero_idxs)
