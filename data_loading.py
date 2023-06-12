import json, jsonlines
import re
import random
from pprint import pprint
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional
from types import MethodType 

from fire import Fire
from pydantic import BaseModel, Extra
from tqdm import tqdm

class ExtraModel(BaseModel, extra="allow"):
    pass

class Sample(BaseModel):
    id: str = ""
    text: str = ""


class CaptionSample(Sample):
    image_path: str


class CocoData(ExtraModel):
    train_image_path: Path = Path("data/COCO/train2014/")
    val_image_path: Path = Path("data/COCO/val2014")

    def load_image_id2path(self) -> dict:
        image_name_to_path = {}
        for child in self.train_image_path.iterdir():
            if child.suffix == '.jpg':
                child_name = child.name.split('_')[-1]
                image_name_to_path[child_name] = child.resolve()
        for child in self.val_image_path.iterdir():
            if child.suffix == '.jpg':
                child_name = child.name.split('_')[-1]
                image_name_to_path[child_name] = child.resolve()
        self.map_image_to_path = image_name_to_path

    def load(self) -> List[Dict]:
        self.load_image_id2path()
        with open(self.path_raw, "r") as f:
            raw = json.load(f)
        return raw


class CocoCaptionData(CocoData):
    task_name: str = "image_caption"
    path_raw: str = "data/COCO/COCO/annotations/captions_train2017.json"
    prompts: List[str] = [
        "How can we describe this photo?",
        "What is the description of this image?",
        "Generate a caption for this photo.",
        "Describe this image.",
        "Produce an explanation of the picture.",
    ]

    def preprocess_raw(self) -> List[dict]:
        annotations = self.load()
        data = []

        for sample in tqdm(annotations):
            image_name = str(sample['image_id']).zfill(12) + '.jpg'
            if image_name in self.map_image_to_path:
                sample = CaptionSample(
                    id = sample['id'],
                    text = sample['caption'],
                    image_path = str(self.map_image_to_path[image_name].resolve())
                )
                data.append(sample)

        return data

class VQASample(Sample):
    image_path: str = ""
    answer: Union[str, List[str]] = []

class OKVQAData(CocoData):
    task_name: str = "vqa"
    path_raw: str = "data/okvqa/annotations/okvqa_train.json"

    def preprocess_raw(self) -> List[dict]:
        raw = self.load()
        data = []
        for sample in tqdm(raw):
            image_name = sample['image'].split('_')[-1]
            if image_name in self.map_image_to_path:
                sample = VQASample(
                    id = sample["question_id"],
                    text = sample["question"],
                    image_path = str(self.map_image_to_path[image_name].resolve()),
                    answer = sample['answer'],
                )
                data.append(sample)
        return data

class AOKVQASample(VQASample):
    options: List[str]
    rationales: List[str]

class AOKVQAData(CocoData):
    task_name: str = 'vqa'
    path_raw: str = "data/aokvqa/annotations/aokvqa_v1p0_train.json"

    def preprocess_raw(self) -> List[dict]:
        data = []
        raw = self.load()
        for i, sample in enumerate(tqdm(raw)):
            image_name = sample["image"].split('_')[-1]
            if image_name in self.map_image_to_path:
                sample = AOKVQASample(
                    id = i,
                    text = sample["question"],
                    image_path = str(self.map_image_to_path[image_name].resolve()),
                    answer = sample["choices"][sample["correct_choice_idx"]],
                    options = sample["choices"],
                    rationales = sample["rationales"],
                )
                data.append(sample)

        return data

class VideoSample(Sample):
    video_path: str

class WebvidsData(ExtraModel):
    task_name: str = "video caption"
    path_raw: str = "/mnt/data_16tb/navo/trimera/datasets/webvids/results_2M_train.jsonl"
    train_video_path: Path = Path("/mnt/data_16tb/navo/trimera/datasets/webvids/data/train")

    def load_video_id2path(self) -> dict:
        video_name_to_path = {}
        for child in tqdm(self.train_video_path.iterdir()):
            if child.suffix == '.mp4':
                child_name = child.name
                video_name_to_path[child_name] = child.resolve()
        self.map_video_to_path = video_name_to_path

    def load(self) -> List[Dict]:
        self.load_video_id2path()
        raw_data = []
        with jsonlines.open(self.path_raw) as raw:
            for sample in raw:
                raw_data.append(sample)
        return raw_data

    def preprocess_raw(self):
        raw = self.load()
        data = []

        for i, sample in enumerate(tqdm(raw)):
            video_name = sample["fname"].split('/')[-1]
            if video_name in self.map_video_to_path:
                sample = VideoSample(
                    id = i,
                    text = sample["caption"],
                    video_path = str(self.map_video_to_path[video_name].resolve()),
                )
                data.append(sample)
        return data




def test_model(name: str):
    if name == 'cococaption':
        dataset = CocoCaptionData()
    elif name == 'okvqa':
        dataset = OKVQAData()
    elif name == "aokvqa":
        dataset = AOKVQAData()
    elif name == "webvids":
        # This will take around 4 mins
        dataset = WebvidsData()
    else:
        raise ValueError("dataset currently not included")

    data = dataset.preprocess_raw()
    print(f'dataset size:{round(len(data)/1e6, 2)} M')
    pprint(random.choices(data, k=5))


"""
TODO:
"""

if __name__ == "__main__":
    Fire()
