#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create serialized data for dialogue state tracking."""

import argparse
from collections import  defaultdict
from deepdiff import DeepDiff
from pprint import pprint
import re
import json
import os
import random
import time

from tqdm import tqdm

from utils import flatten_ds, get_logger, get_schema


logger = get_logger(__name__)
random.seed(1007)

GREETINGS = ["hello .",
             "what can i do for you ?",
             "hello , what can i do for you ?",
             "is there anything i can do for you ?"]


def setup_args():
    """Setup arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="multiwoz", choices=["multiwoz", "woz"])
    parser.add_argument("--data_type", type=str, choices=["train", "dev", "test"], required=True)
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    return args

# data_file: 
def main(args):
    """Main function."""
    with open(args.data_file, "r") as fin:
        data = json.load(fin)
        logger.info(f"Load dataset from `{args.data_file}`")

    schema = get_schema(args.dataset)

    # get description from schmea.json
    # https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.2/schema.json
    desc_path = "./projects/AG-DST/data/"
    desc_path = desc_path + "schema.json"
    description_schema = json.load(open(desc_path, "r"))
    logger.info(f"Load description from `{desc_path}`")

    frame_idxs = {"train": 0, "taxi":1, "bus":2, "police":3, "hotel":4, "restaurant":5, "attraction":6, "hospital":7}
    domain_desc_flag = True # To append domain descriptions or not 
    slot_desc_flag = True  # To append slot descriptions or not 
    PVs_flag = True # for categorical slots, append possible values as suffix

    # data: "./projects/AG-DST/data/${dataset}/processed/${data_type}_data_withneg.json"
    empty_ds_seq = "<ds/> " + " ".join(flatten_ds({}, schema)) + " </ds>"
    ds_labels = []
    ds_seqs = []
    # dial 은 utterance, dialogue state, label 등 다양한 정보 포함
    for dial_id, dial in tqdm(data.items(), desc="Dialogue"): # dial_id example : MUL0001.json
        dial_utt_seqs = process_utt(dial)
        # dial_ds_seqs: ds dial_ds_labels를 sequence 로 만듬
        # dial_ds_labels와 dial_ds_seqs 두개를 들고 있는 이유
        # -> dial_ds_seqs 는 이전 previoust state를 의미하는 입력으로 들어감
        # -> dial_ds_labels 는 ds_labels 에 쌓여서 파일로 저장됨.
        (dial_ds_labels, dial_ds_seqs), (_, dial_neg_ds_seqs) = process_ds(dial, schema)
        ds_labels.extend(dial_ds_labels)

        # find corresponding domain and slot for description 
        # (dial_domain, dial_slot) , (dial_neg_domain, dial_neg_slot) = find_desc(dial,schema)
        find_desc(dial,schema)
        time.sleep(100)
        print("\n\ncheck\n\n")
        if args.data_type in ("train", "dev"):
            # concatenate utterance and dialogue state for training: cur_utt + prev_ds -> cur_ds
            for idx, (turn_utt_seq, turn_ds_seq, turn_neg_ds_seq) in \
                enumerate(zip(dial_utt_seqs, dial_ds_seqs, dial_neg_ds_seqs)):
            # for idx, (turn_utt_seq, turn_ds_seq, turn_neg_ds_seq, turn_domain, turn_slot, turn_neg_domain, turn_neg_slot) in \
            #     enumerate(zip(dial_utt_seqs, dial_ds_seqs, dial_neg_ds_seqs, dial_domain, dial_slot,dial_neg_domain, dial_neg_slot)):
                # basic generation
                if idx == 0:
                    prev_turn_ds_seq = empty_ds_seq
                    
                
                
                ds_seqs.append(f"<gen/> {turn_utt_seq} [SEP] {prev_turn_ds_seq} </gen>\x010\t{turn_ds_seq}\x010")
                # print("prev_turn_ds_seq: " + prev_turn_ds_seq + "\n")
                # print("prev_turn_ds_seq len:" + str(len(prev_turn_ds_seq)))
                # print("turn_ds_seq: " + turn_ds_seq)
                # print("turn_ds_seq len:" + str(len(turn_ds_seq)))
                # time.sleep(100)
                
                # amending generation
                ds_seqs.append(f"<amend/> {turn_utt_seq} [SEP] {turn_neg_ds_seq} </amend>\x010\t{turn_ds_seq}\x010")
                prev_turn_ds_seq = turn_ds_seq

        elif args.data_type == "test":
            # save utterance for inference
            for idx, turn_utt_seq in enumerate(dial_utt_seqs):
                ds_seqs.append(f"{dial_id}\t{2 * idx}\t{turn_utt_seq}")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    label_path = os.path.join(args.save_path, f"{args.data_type}_labels.json")
    with open(label_path, "w") as fout:
        json.dump(ds_labels, fout, indent=2)
        logger.info(f"Save DS labels to `{label_path}`")

    seq_path = os.path.join(args.save_path, f"{args.data_type}_seq.tsv")
    with open(seq_path, "w") as fout:
        if args.data_type in ("train", "dev"):
            ds_seqs.insert(0, "src\ttgt")
        else:
            ds_seqs.insert(0, "dial_id\tturn_idx\tutts")
        fout.write("\n".join(ds_seqs))
        logger.info(f"Save sequence to `{seq_path}`")


def process_utt(dial):
    """Convert utterance into sequence."""
    greeting = random.choice(GREETINGS)
    dial_utts = [greeting] + [turn["processed_text"] for turn in dial["log"]]
    dial_utt_seqs = []
    for idx in range(len(dial_utts)):
        if idx % 2 != 0:
            # user turn
            last_two_utts = dial_utts[idx - 1: idx + 1]
            dial_utt_seqs.append(f"<utt/> <sys> {last_two_utts[0]}\x010 [SEP] <user> {last_two_utts[1]} </utt>\x011")
    return dial_utt_seqs


def process_ds(dial, schema):
    """Convert dialogue state into sequence."""
    dial_ds_labels, dial_neg_ds_labels = extract_ds(dial)
    dial_ds_seqs = []
    dial_neg_ds_seqs = []

    # ground DS
    # ds label을 sequence로 만들고 special token 추가.
    for turn_ds_label in dial_ds_labels:
        turn_ds_seq_ls = flatten_ds(turn_ds_label, schema)
        # add special token
        turn_ds_seq = "<ds/> " + " ".join(turn_ds_seq_ls) + " </ds>"
        dial_ds_seqs.append(turn_ds_seq)
    # negative DS
    for turn_neg_ds_label in dial_neg_ds_labels:
        turn_neg_ds_seq_ls = flatten_ds(turn_neg_ds_label, schema)
        # add special token
        turn_neg_ds_seq = "<ds/> " + " ".join(turn_neg_ds_seq_ls) + " </ds>"
        dial_neg_ds_seqs.append(turn_neg_ds_seq)

    return (dial_ds_labels, dial_ds_seqs), (dial_neg_ds_labels, dial_neg_ds_seqs)


def extract_ds(dial):
    """Extract dialogue state."""
    dial_ds_labels = []
    dial_neg_ds_labels = []

    for turn_idx, turn in enumerate(dial["log"]):
        if turn_idx % 2 == 0:
            # user turn
            continue
        # ground DS 
        # 정답 dialogue state 인듯
        turn_ds_labels = defaultdict(lambda: defaultdict(dict))
        # processed_metadata는 모든 domain에 해당하는 slot 칸이 정의되있음
        # 거기서 각 도메인별 각 슬롯별로 label 값 저장.
        for dom, dom_ds in turn["processed_metadata"].items():
            for slot_type, slot_vals in dom_ds.items():
                for slot, vals in slot_vals.items():
                    if slot != "booked" and len(vals) > 0:
                        turn_ds_labels[dom][slot_type][slot] = vals
        dial_ds_labels.append(turn_ds_labels)
        # negative DS
        # negative metadata에는 따로 있어서 이걸 따로함.
        if "negative_metadata" in turn:
            turn_neg_ds_labels = defaultdict(lambda: defaultdict(dict))
            for dom, dom_ds in turn["negative_metadata"].items():
                for slot_type, slot_vals in dom_ds.items():
                    for slot, vals in slot_vals.items():
                        if slot != "booked" and len(vals) > 0:
                            turn_neg_ds_labels[dom][slot_type][slot] = vals
            dial_neg_ds_labels.append(turn_neg_ds_labels)

    return dial_ds_labels, dial_neg_ds_labels

def find_desc(dial,schema):
    """input dialogue state 와 label dialogue state 를 비교하여
    어떤 domain-slot pair description을 추가해야하는지 결정"""
    """input dialogue state: prev_turn_"""
    dial_ds_labels, dial_neg_ds_labels = extract_ds(dial)
    
    # return domain_slot_finding(dial_ds_labels), domain_slot_finding(dial_neg_ds_labels)
    domain_slot_finding(dial_ds_labels)

def domain_slot_finding(dial_ds_labels):
    domain = []
    slot = [["NONE"]]
    
    prev_turn_ds_label = defaultdict(lambda: defaultdict(dict))
    
    for turn_idx, turn_ds_label in enumerate(dial_ds_labels):
        domain_part = []
        # find domain name in current turn
        for domain_name in turn_ds_label:
            domain_part.append(domain_name)
        domain.append(domain_part)

        # skip slot name finding in initial turn
        if (turn_idx == 0):
            prev_turn_ds_label = turn_ds_label
            continue

        # compare prev ds and current ds
        # To compare complex dictionaries. use deepdiff python library (pip install deepdiff)
        if (prev_turn_ds_label != turn_ds_label):
            for prev_key1, cur_key1 in zip(prev_turn_ds_label, turn_ds_label):
                if (prev_key1 == cur_key1):
                    diff = DeepDiff(prev_turn_ds_label[prev_key1], turn_ds_label[cur_key1],verbose_level=2)
                
                    # dictionary_item_added
                    if "dictionary_item_added" in diff:
                        # many slot_name
                        if len(diff["dictionary_item_added"]) > 1:
                            slot_part = []    
                            for slot_name in diff["dictionary_item_added"]:
                                slot_part.extend(extract_slot_from_deepdiff(slot_name,diff["dictionary_item_added"]))
                            slot.append(slot_part)
                        else:
                            # only one slot_name 
                            for slot_name in diff["dictionary_item_added"]:
                                slot.append(extract_slot_from_deepdiff(slot_name,diff["dictionary_item_added"]))
                    
                    # values_changed
                    if "values_changed" in diff:
                        # many slot_name
                        if len(diff["values_changed"]) > 1:
                            slot_part = []    
                            for slot_name in diff["values_changed"]:
                                slot_part.extend(extract_slot_from_deepdiff(slot_name,diff["values_changed"]))
                            slot.append(slot_part)
                        else:
                            # only one slot_name 
                            for slot_name in diff["values_changed"]:
                                slot.append(extract_slot_from_deepdiff(slot_name,diff["values_changed"]))
                                
                    # dictionary_item_removed
                    if "dictionary_item_removed" in diff:
                        # many slot_name
                        if len(diff["dictionary_item_removed"]) > 1:
                            slot_part = []    
                            for slot_name in diff["dictionary_item_removed"]:
                                slot_part.extend(extract_slot_from_deepdiff(slot_name,diff["dictionary_item_removed"]))
                            slot.append(slot_part)
                        else:
                            # only one slot_name 
                            for slot_name in diff["dictionary_item_removed"]:
                                slot.append(extract_slot_from_deepdiff(slot_name,diff["dictionary_item_removed"]))
                                                
                    
                    pprint(diff)

                else:
                    slot.append(["NONE"])
        else:
            slot.append(["NONE"])

        print(prev_turn_ds_label)
        print(turn_ds_label)
        print("\n")
        prev_turn_ds_label = turn_ds_label
        
    pprint(slot, indent=1)
    pprint(domain, indent=1)
    print(len(domain))
    return domain, slot
        
def extract_slot_from_deepdiff(slot_name,diff_specific):
    """extract the slot name from the result of deepdiff library"""
    sub_slot = []
    str = ""
    if (slot_name.count("[") > 1):
        slot_name = slot_name.replace("'","")
        for idx, val in enumerate(slot_name):
            if (val == "["):
                for j in slot_name[idx+1:]:
                    if (j == "]"): break
                    str += j
                if (str == "semi"): str = ""    
        # remove digit in slot_name and append
        sub_slot.append(''.join([i for i in str if not i.isdigit()]))
        
    else:
        extra_slot_name = list(diff_specific[slot_name].keys())
        complete_slot_name = ""
        # ext_slot_list = []
        for ext_slot in extra_slot_name:
            str = ""
            complete_slot_name = (slot_name + "[" +  ext_slot + "]").replace("'","")
            for idx, val in enumerate(complete_slot_name):
                if (val == "["):
                    for j in complete_slot_name[idx+1:]:
                        if (j == "]"): break
                        str += j
                    if (str == "semi"): str = ""
            # remove digit in slot name and append
            # ext_slot_list.append(''.join([i for i in str if not i.isdigit()]))
            sub_slot.append(''.join([i for i in str if not i.isdigit()]))
        # sub_slot.append(ext_slot_list)

    return sub_slot


if __name__ == "__main__":
    args = setup_args()
    main(args)
