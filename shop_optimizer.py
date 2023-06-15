import json

from dataclasses import dataclass
from aenum import IntFlag, auto
from functools import cache
from itertools import combinations
from typing import List, Set

import numpy as np
import pandas as pd


"""
ShopOptimizer loads all available shop data and hero data with the goal of finding the following information:

What is the maximum amount of silver that we can earn in the shop?
As a secondary criteria, can we also maximize the amount of gold that we can earn?

We need to:
1. Load all shop data as a JSON, and index shop data as a map, where key=name, value = (silver, gold), buffs
2. Load all hero data as a JSON, and index hero data as a map, where key=name, value = buffs
"""

class BuffType(IntFlag):
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
class ShopItem:
    name: str
    income: Income
    buff_types: BuffType

@dataclass
class Buff:
    buff: BuffType
    value : float

@dataclass
class Hero:
    name: str
    buffs: tuple[Buff]



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
                buff_types=BuffType['|'.join(data['types'])]
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
                    Buff(buff=BuffType[buff_type], value=value)
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
        max_soln, combos = Income(silver=0, gold=0, powder=0), ()
        for hero_combo in self.hero_combos:
            for item_combo in self.item_combos:
                res = self.__eval_multiple(hero_combo, item_combo)
                if res > max_soln:
                    max_soln = res
                    combos = (hero_combo, item_combo)
        
        return max_soln, combos
    
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

        



opt = ShopOptimizer()
# print(opt.hero_dict)
# print(opt.shop_dict)

avia = opt.hero_dict['avia']
elf_necklace = opt.shop_dict['elf_necklace']
# print(elf_necklace)
# print(opt.eval_once('avia', 'elf_necklace'))

max_soln, combos = opt.run()
print(max_soln)
print(combos)