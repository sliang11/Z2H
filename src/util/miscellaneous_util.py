from typing import Union
import argparse

def gen_conf_dict(args: argparse.Namespace, excluded_keys: Union[list, tuple] = None):
    return {k: v for k, v in vars(args).items() if k not in (excluded_keys if excluded_keys is not None else [])}


def pack_vars_to_dict(**kwargs):
    return kwargs


def nested_dict_to_nested_list(dict_to_convert, info_list, current_depth=0):
    if current_depth >= len(info_list) or not isinstance(dict_to_convert, dict):
        return dict_to_convert

    instruction = info_list[current_depth]

    if isinstance(instruction, list):
        return [
            nested_dict_to_nested_list(dict_to_convert[k], info_list, current_depth + 1)
            for k in instruction
        ]
    else:
        target_sub_dict = dict_to_convert[instruction]
        return nested_dict_to_nested_list(target_sub_dict, info_list, current_depth + 1)

