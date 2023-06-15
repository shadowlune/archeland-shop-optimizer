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

We need to:
1. Load all shop data as a JSON, and index shop data as a map, where key=name, value = (silver, gold), buffs
2. Load all hero data as a JSON, and index hero data as a map, where key=name, value = buffs
"""

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
class ShopItem:
    name: str
    income: Income
    buff_types: BuffFlag

@dataclass
class Buff:
    buff: BuffFlag
    value : float

@dataclass
class BuffV2:
    buff: BuffEnum
    value : float

@dataclass
class Hero:
    name: str
    buffs: tuple[Buff]

@dataclass
class HeroV2:
    name: str
    buffs: dict[BuffEnum]




class ShopOptimizer:
    def __init__(self):
        self.__load_data()

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
                # Had to use `aenum` package here because `enum` from stdlib doesn't support associative '|' bitwise or operations
                buff_types=BuffFlag['|'.join(data['types'])]
            )
            for data in shop_data
        }
        shop_item_names = list(self.shop_dict.keys())
        self.item_combos = list(combinations(shop_item_names, r=5))

        with open('hero_shop_modifiers.json') as f:
            hero_data = json.load(f)
        self.hero_dict = {
            data['name']: Hero(
                name = data['name'],
                buffs = (
                    Buff(buff=BuffFlag[buff_type], value=value)
                    for buff_type, value in data['buffs'].items()
                )
            )
            for data in hero_data
        }
        hero_names = list(self.hero_dict.keys())
        self.hero_combos = list(combinations(hero_names, r=5))


    
    def run(self):
        """
        Generate all possible candidate optimizations, chuck into evaluator function.
        """
        max_profit, combos = Income(silver=0, gold=0, powder=0), ()
        for hero_combo in self.hero_combos:
            for item_combo in self.item_combos:
                res = self.__eval_multiple(hero_combo, item_combo)
                if res > max_profit:
                    max_profit = res
                    combos = (hero_combo, item_combo)
        
        return max_profit, combos
    
    def __eval_multiple(self, heroes: List[str], items: List[str]) -> Income:
        return sum(
            self.__eval_once(hero, item)
            for hero in heroes
            for item in items
        )  

    @cache
    def __eval_once(self, hero_name: str, item_name: str) -> Income:
        """
        Evaluate the result of a hero-item pair.
        """
        hero = self.hero_dict[hero_name]
        item = self.shop_dict[item_name]

        multiplier = 1 + sum([
            buff.value
            for buff in hero.buffs
            if buff.buff in item.buff_types
        ])
        return item.income * multiplier



"""
Version 2 of the shop optimizer.
No longer tries to find all combinations of heroes and items.
Instead, we generate all possible combination of heroes, and fix the items as a weighted matrix of size: (num_items, num_buffs)
"""
class ShopOptimizerV2:
    def __init__(self):
        self.__load_data()

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

        self.shop_matrix = np.array([[
                shop_item.income.silver 
                if buff in shop_item.buff_types
                else 0 
                for buff in BuffEnum
            ]
            for shop_item in self.shop_dict.values()
        ])
        self.shop_matrix = np.hstack((
            np.array([shop_item.income.silver for shop_item in self.shop_dict.values()]).reshape(-1,1),
            self.shop_matrix
        ))
        print("Matrix shape:")
        print(self.shop_matrix.shape)
        print(f"Matrix: {self.shop_matrix}") 

        self.shop_item_names = list(self.shop_dict.keys())

        with open('hero_shop_modifiers.json') as f:
            hero_data = json.load(f)

        #  -> { BuffV2("magic", 0.42), BuffV2("shell", 0.3)}
                # magic      shell        gourmet      
        #  hero1  0.42       0.3           0       ...
        #  hero2  0           0            0.2     ... 
        #  hero3 
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
        self.hero_matrix = np.array([[
                hero.buffs[buff]
                if buff in hero.buffs.keys()
                else 0 
                for buff in BuffEnum
            ]
            for hero in self.hero_dict.values()
        ])

        self.hero_names = list(self.hero_dict.keys())
        # For our numpy array slicing to work correctly, we need to spawn index masks like so
        self.hero_combos = [list(mask) for mask in combinations(range(len(self.hero_names)), r=5)]
                   
        #  -> { BuffV2("magic", 0.42), BuffV2("shell", 0.3)}
                # magic      shell        gourmet      
        #  hero1  0.42       0.3           0       ...
        #  hero2  0           0            0.2     ... 
        #  hero3 
        #  hero16
        #  hero17
        # .... 
                                        # magic     shell        gourmet                  
        # hero_combined(1,2,3,16,17)      0.42        0.3           0.2   
        # -> base   magic shell gourmet
        #    1       .42      .3    .2  .....  ->

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
        hero_vector = np.insert(hero_vector, 0, 1, axis=0) # Prepend 1 to the vector to account for the base income

        # print("Hero vector:")
        # print(hero_vector)
        # print('----')
        # print(self.shop_matrix)
        # print('----- Results')
        prod = self.shop_matrix @ hero_vector
        # print(prod)
        # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        top_six_indices = np.argpartition(prod, -6)[-6:]
        profit = np.sum(prod[top_six_indices])
        return profit, list(top_six_indices)

opt = ShopOptimizerV2()
max_profit, item_idxs, hero_idxs = opt.run()
print(f"Max Profit: {max_profit}")
print(f"Item indices: {item_idxs}")
print(f"Hero indices: {hero_idxs}")

heroes = [opt.hero_names[idx] for idx in hero_idxs]
items = [opt.shop_item_names[idx] for idx in item_idxs]

print(f"Hero names: {heroes}")
print(f"Item names: {items}")



# print(opt.hero_dict)
# print(opt.shop_dict)

# avia = opt.hero_dict['avia']
# elf_necklace = opt.shop_dict['elf_necklace']
# print(elf_necklace)
# print(opt.eval_once('avia', 'elf_necklace'))

# print(max_profit)
# print(combos)