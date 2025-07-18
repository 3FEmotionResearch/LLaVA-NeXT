# Source from https://github.com/zeroQiaoba/AffectGPT/blob/bf68d98fc4b6709ba46b29cf27c2dce6fd25e888/AffectGPT/my_affectgpt/evaluation/wheel.py#L1 .
"""
# How to run
cd $REPO_ROOT

python -m scripts.video.eval.mercaptionplus_metrics_cal_my \
  --dataset_path /home/paperspace/Downloads/affectgpt-dataset-mini100 \
  --predictions_path /home/paperspace/ReposPublic/25-07-14-LLaVA-NeXT/work_dirs/emotion_detect_infer_outputs/LLaVA-NeXT-Video-7B-DPO_vicuna_v1_frames_8_stride_2/predictions.json
"""

import argparse
import os
import pandas as pd
import numpy as np
import re
import glob
import json
from pathlib import Path

# Import shared data loader
from scripts.video.data_loader import create_enhanced_merged_dataset


# S: Constants.
def get_project_root():
    """Find the project root by looking for .git directory"""
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / ".git").exists():
            return parent
    raise FileNotFoundError("Project root (with .git directory) not found. Make sure you're running this from within a git repository.")


PROJECT_ROOT = get_project_root()
EMOTION_WHEEL_ROOT = str(PROJECT_ROOT / "scripts" / "video" / "eval" / "emotion_wheel")
# E: Constants.


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Calculate emotion detection metrics using emotion wheel")
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the MERCaptionPlus dataset folder"
    )
    
    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
        help="Path to the predictions JSON file"
    )
    
    parser.add_argument(
        "--emotion_label_field",
        type=str,
        default="emotion_labels",
        help="Field name in predictions JSON that contains emotion labels"
    )
    
    parser.add_argument(
        "--level",
        type=str,
        default="level1",
        choices=["level1", "level2"],
        help="Emotion wheel level for evaluation"
    )
    
    return parser.parse_args()


# S: Helpers.
# "['轻松', '愉快', '幽默', '自嘲']" => ['轻松', '愉快', '幽默', '自嘲']
def string_to_list(str):

    if type(str) == np.ndarray:
        str = str.tolist()

    # 如果本身就是list了，那么就不需要其他操作了
    if isinstance(str, list):
        return str

    if str == "":
        str = []
    elif pd.isna(str):
        str = []
    else:
        if str[0] == "[":
            str = str[1:]
        if str[-1] == "]":
            str = str[:-1]
        str = [item.strip() for item in re.split("['\",]", str) if item.strip() not in ["", ","]]
    return str


# 从csv中读取特定的key对应的值
def func_read_key_from_csv(csv_path, key):
    values = []
    df = pd.read_csv(csv_path)
    # for _, row in df.iterrows():
    for _, row in df.iterrows():
        if key not in row:
            values.append("")
        else:
            value = row[key]
            try:
                if pd.isna(value):
                    value = ""
            except (TypeError, ValueError):
                # Handle cases where pd.isna returns array-like objects
                if hasattr(value, "__iter__") and not isinstance(value, str):
                    value = ""
            values.append(value)
    return values


def read_format2raws():
    format2raws = {}

    format_path = os.path.join(EMOTION_WHEEL_ROOT, "format.csv")
    raws = func_read_key_from_csv(format_path, "name")
    formats = func_read_key_from_csv(format_path, "format")
    for raw, format in zip(raws, formats):

        # 1. 建立 format 与 raw 之间的映射
        format = string_to_list(format)
        for format_item in format:
            if format_item not in format2raws:
                format2raws[format_item] = []
            format2raws[format_item].append(raw)

        # 2. 建立 raw 与 raw 之间的映射
        if raw not in format2raws:
            format2raws[raw] = []
        format2raws[raw].append(raw)
    print(f"format2raws: {len(format2raws)}")
    return format2raws


#############################################
###### 从所有 emotion wheel 中读取情感词 ######
#############################################
# read xlsx and convert it into map format
def read_wheel_to_map(xlsx_path):
    store_map = {}
    level1, level2, level3 = "", "", ""

    df = pd.read_excel(xlsx_path)
    for _, row in df.iterrows():
        row_level1 = row["level1"]
        row_level2 = row["level2"]
        row_level3 = row["level3"]

        # update level1, level2, level3
        if not pd.isna(row_level1):
            level1 = row_level1
        if not pd.isna(row_level2):
            level2 = row_level2
        if not pd.isna(row_level3):
            level3 = row_level3

        # store into store_map [存储前需要经过预处理]
        level1 = level1.lower().strip()
        level2 = level2.lower().strip()
        level3 = level3.lower().strip()
        if level1 not in store_map:
            store_map[level1] = {}
        if level2 not in store_map[level1]:
            store_map[level1][level2] = []
        store_map[level1][level2].append(level3)
    return store_map


# 所有 emotion wheel 合并，可以生成 248 个候选单词 => 247 候选标签了，跟我修正 fearful 有关系
def convert_all_wheels_to_candidate_labels():
    candidate_labels = []
    for xlsx_path in glob.glob(EMOTION_WHEEL_ROOT + "/wheel*.xlsx"):
        store_map = read_wheel_to_map(xlsx_path)
        for level1 in store_map:
            for level2 in store_map[level1]:
                level3 = store_map[level1][level2]
                candidate_labels.append(level1)
                candidate_labels.append(level2)
                candidate_labels.extend(level3)
    candidate_labels = list(set(candidate_labels))
    return candidate_labels


"""
Totally, we can generate 253 emotion-wheel labels
"""


# 1. 读取候选词的同义词形式
# label2wheel: 实现所有标签 -> emotion wheel 中的标签
def read_candidate_synonym_onerun(runname="run1"):

    ## read candidate labels
    wheel_labels = convert_all_wheels_to_candidate_labels()

    ## gain mapping
    label2wheel = {}
    synonym_path = os.path.join(EMOTION_WHEEL_ROOT, "synonym.xlsx")
    df = pd.read_excel(synonym_path)
    for _, row in df.iterrows():

        # 建立 self-mapping
        raw = str(row[f"word_{runname}"]).strip().lower()
        assert raw in wheel_labels, f"error in {raw}"  # check openai returns
        if raw not in label2wheel:
            label2wheel[raw] = []
        label2wheel[raw].append(raw)

        # 建立 synonyms -> raw 映射
        synonyms = row[f"synonym_{runname}"]
        synonyms = string_to_list(synonyms)
        for synonym in synonyms:
            synonym = str(synonym).strip().lower()
            if synonym not in label2wheel:
                label2wheel[synonym] = []
            label2wheel[synonym].append(raw)
    return label2wheel


# func: 合并两个 map 合并，map中的元素的list的
def func_merge_map(map1, map2):
    all_items = list(map1.keys()) + list(map2.keys())

    merge_map = {}
    for item in all_items:
        if item in map1 and item in map2:
            value = list(set(map1[item] + map2[item]))
        elif item not in map1 and item in map2:
            value = map2[item]
        elif item in map1 and item not in map2:
            value = map1[item]
        merge_map[item] = value
    return merge_map


def read_candidate_synonym_merge():
    mapping_run1 = read_candidate_synonym_onerun("run1")
    mapping_run2 = read_candidate_synonym_onerun("run2")
    mapping_run3 = read_candidate_synonym_onerun("run3")
    mapping_run4 = read_candidate_synonym_onerun("run4")
    mapping_run5 = read_candidate_synonym_onerun("run5")
    mapping_run6 = read_candidate_synonym_onerun("run6")
    mapping_run7 = read_candidate_synonym_onerun("run7")
    mapping_run8 = read_candidate_synonym_onerun("run8")
    mapping_merge = func_merge_map(mapping_run1, mapping_run2)
    mapping_merge = func_merge_map(mapping_merge, mapping_run3)
    mapping_merge = func_merge_map(mapping_merge, mapping_run4)
    mapping_merge = func_merge_map(mapping_merge, mapping_run5)
    mapping_merge = func_merge_map(mapping_merge, mapping_run6)
    mapping_merge = func_merge_map(mapping_merge, mapping_run7)
    mapping_merge = func_merge_map(mapping_merge, mapping_run8)
    print(f"label number: {len(mapping_merge)}")
    return mapping_merge


#########################################################################
######## 采用得到的mapping，去度量 gt and pred openset 之间的重叠度 ########
## => 所有评价指标，都是把不在词表中的元素直接剔除掉 [统一化处理，方便后续比较]
##########################################################################
def func_get_name2reason(reason_root):
    name2reason = {}
    for reason_npy in glob.glob(reason_root + "/*.npy"):
        name = os.path.basename(reason_npy)[:-4]
        reason = np.load(reason_npy).tolist()
        name2reason[name] = reason
    return name2reason


def func_get_wheel_cluster(wheel="wheel1", level="level1"):
    xlsx_path = os.path.join(EMOTION_WHEEL_ROOT, f"{wheel}.xlsx")
    emotion_wheel = read_wheel_to_map(xlsx_path)
    wheel_map = {}

    # 1. 所有聚类到level1
    if level == "level1":
        for level1 in emotion_wheel:
            wheel_map[level1] = level1
            for level2 in emotion_wheel[level1]:
                wheel_map[level2] = level1
                for level3 in emotion_wheel[level1][level2]:
                    wheel_map[level3] = level1

    # 2. 全部聚类到level2
    elif level == "level2":
        for level1 in emotion_wheel:
            wheel_map[level1] = sorted(emotion_wheel[level1])[0]  # level1 映射到一个固定的 level2 上
            for level2 in emotion_wheel[level1]:
                wheel_map[level2] = level2
                for level3 in emotion_wheel[level1][level2]:
                    wheel_map[level3] = level2
    return wheel_map


# 计算一个 （gt_root, openset_root）在 metric 下的重叠率
# => 修改代码：支持不同格式的输入文件，并且支持只计算样本子集的结果
# => 修改代码：支持外部读取 format_mapping 和 raw_mapping
def calculate_openset_overlap_rate(name2gt=None, name2pred=None, process_names=None, metric="case1", format_mapping=None, raw_mapping=None, inter_print=True):

    # Assert that required parameters are not None
    assert name2gt is not None, "name2gt cannot be None"
    assert name2pred is not None, "name2pred cannot be None"

    # process_names => (whole) / (subset)
    if process_names is None:
        process_names = list(name2gt.keys())

    # read all mapping
    if format_mapping is None:
        format_mapping = read_format2raws()  # level3 -> level2
    if raw_mapping is None:
        raw_mapping = read_candidate_synonym_merge()  # level2 -> level1
    if metric.startswith("case3"):  # level1 -> cluster center
        _, wheelname, levelname = metric.split("_")
        wheel_map = func_get_wheel_cluster(wheelname, levelname)
    else:
        wheel_map = None

    # calculate (accuracy, recall) two values
    accuracy, recall = [], []
    for name in process_names:

        # 删除 gt and pred 中的同义词
        gt = string_to_list(name2gt[name])
        gt = [item.lower().strip() for item in gt]
        gt = set(func_map_label_to_synonym(gt, format_mapping, raw_mapping, wheel_map, metric))

        pred = string_to_list(name2pred[name])
        pred = [item.lower().strip() for item in pred]
        pred = set(func_map_label_to_synonym(pred, format_mapping, raw_mapping, wheel_map, metric))

        if len(gt) == 0:
            continue
        if len(pred) == 0:
            accuracy.append(0)
            recall.append(0)
        else:
            accuracy.append(len(gt & pred) / len(pred))
            recall.append(len(gt & pred) / len(gt))
    if inter_print:
        print("process number (after filter): ", len(accuracy))
    avg_accuracy, avg_recall = np.mean(accuracy), np.mean(recall)
    if inter_print:
        print(f"avg acc: {avg_accuracy} avg recall: {avg_recall}")
    return avg_accuracy, avg_recall


# 计算只依赖于 format_mapping 下的结果
def func_backward_case1(label, format_mapping, raw_mapping=None, wheel_map=None):
    if label not in format_mapping:
        return ""

    stage1_labels = format_mapping[label]
    assert isinstance(stage1_labels, list)
    stage1_unique = sorted(stage1_labels)[0]

    return stage1_unique


# 核心是保证 backward 过程中的唯一性
def func_backward_case2(label, format_mapping, raw_mapping, wheel_map=None):
    if label not in format_mapping:
        return ""

    stage1_labels = format_mapping[label]
    assert isinstance(stage1_labels, list)
    stage1_unique = sorted(stage1_labels)[0]

    stage2_labels = raw_mapping[stage1_unique]
    assert isinstance(stage2_labels, list)
    stage2_unique = sorted(stage2_labels)[0]
    return stage2_unique


## 引入 emotion wheel 进行评价
def func_backward_case3(label, format_mapping, raw_mapping, wheel_map):
    if label not in format_mapping:
        return ""

    level1_whole = []
    for format in format_mapping[label]:
        for raw in raw_mapping[format]:
            level1_whole.append(raw)

    for level1 in sorted(level1_whole):  # 保证了结果唯一性
        if level1 in wheel_map:
            return wheel_map[level1]
    return ""


# metric: ['case1', 'case2', 'case3'] 展示的是一种层级化的聚类结果，从而看出每层的必要性
def func_map_label_to_synonym(mlist, format_mapping, raw_mapping, wheel_map, metric="case1"):
    new_mlist = []
    for label in mlist:
        if metric.startswith("case1"):
            label = func_backward_case1(label, format_mapping)
        if metric.startswith("case2"):
            label = func_backward_case2(label, format_mapping, raw_mapping)
        if metric.startswith("case3"):
            label = func_backward_case3(label, format_mapping, raw_mapping, wheel_map)
        if label == "":
            continue  # 如果找不到 backward 的词，就把他剔除
        new_mlist.append(label)
    return new_mlist


# E: Helpers.

# S: Main funcs.

format_mapping = read_format2raws()  # level3 -> level2
raw_mapping = read_candidate_synonym_merge()  # level2 -> level1


# 功能：input [gt, openset]; output: 12 个 EW-based metric 下的平均结果
def wheel_metric_calculation(name2gt=None, name2pred=None, process_names=None, inter_print=True, level="level1"):
    if level == "level1":
        candidate_metrics = [
            "case3_wheel1_level1",
            "case3_wheel2_level1",
            "case3_wheel3_level1",
            "case3_wheel4_level1",
            "case3_wheel5_level1",
        ]
    elif level == "level2":
        candidate_metrics = [
            "case3_wheel1_level2",
            "case3_wheel2_level2",
            "case3_wheel3_level2",
            "case3_wheel4_level2",
            "case3_wheel5_level2",
        ]

    # 计算每个metric的这个值
    whole_scores = []
    for metric in candidate_metrics:
        precision, recall = calculate_openset_overlap_rate(name2gt=name2gt, name2pred=name2pred, process_names=process_names, metric=metric, format_mapping=format_mapping, raw_mapping=raw_mapping, inter_print=inter_print)
        fscore = 2 * (precision * recall) / (precision + recall)
        whole_scores.append([fscore, precision, recall])
    avg_scores = (np.mean(whole_scores, axis=0)).tolist()
    return avg_scores


# E: Main funcs.

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    print(f"Using dataset path: {args.dataset_path}")
    print(f"Using predictions path: {args.predictions_path}")
    print(f"Using emotion label field: {args.emotion_label_field}")
    print(f"Using evaluation level: {args.level}")

    # Load dataset
    enhanced_merged_by_name = create_enhanced_merged_dataset(args.dataset_path)
    #   name: str -> 'samplenew3_00000891'
    #   openset: str -> '[optimistic, hopeful, encouraged]'
    #   reason: str -> 'In the text, ...'
    #   chinese_subtitle: str -> '花有重开日'
    #   english_subtitle: str -> 'Flowers have a day to bloom again.'

    # Convert enhanced_merged_by_name to name2gt format
    name2gt = {}
    for key, val in enhanced_merged_by_name.items():
        name = key
        openset = val["openset"]
        name2gt[name] = openset

    # Load predictions.json and convert to name2pred format
    with open(args.predictions_path, "r") as f:
        predictions_data = json.load(f)

    name2pred = {}
    for key, val in predictions_data.items():
        name = key
        emotion_labels_str = val[args.emotion_label_field]
        name2pred[name] = emotion_labels_str
    
    # Calculate metrics
    avg_scores = wheel_metric_calculation(name2gt=name2gt, name2pred=name2pred, level=args.level, inter_print=False)

    print(f"  Final Average Scores ({args.level.upper()}):")
    print(f"  F1-Score: {avg_scores[0]:.4f}")
    print(f"  Precision: {avg_scores[1]:.4f}")
    print(f"  Recall: {avg_scores[2]:.4f}")
